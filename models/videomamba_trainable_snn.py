from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.videomamba import RMSNorm, layer_norm_fn, rms_norm_fn
from models.videomamba_clean import CleanVideoMamba


def parse_block_indices(text):
    if isinstance(text, (tuple, list)):
        return tuple(int(value) for value in text)
    values = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    return tuple(values)


def inverse_softplus(x):
    x = torch.clamp(x, min=1e-6)
    return x + torch.log(-torch.expm1(-x))


class TrainableSpikeBase(nn.Module):
    def __init__(
        self,
        num_features,
        threshold_init=1.0,
        threshold_percentile=0.99,
        train_threshold=True,
        signed=True,
        surrogate_alpha=4.0,
        detach_reset=True,
        eps=1e-6,
    ):
        super().__init__()
        self.num_features = int(num_features)
        self.threshold_percentile = float(threshold_percentile)
        self.signed = bool(signed)
        self.surrogate_alpha = float(surrogate_alpha)
        self.detach_reset = bool(detach_reset)
        self.eps = float(eps)
        threshold = torch.full(self.threshold_shape(), float(threshold_init))
        self.log_threshold = nn.Parameter(inverse_softplus(threshold))
        self.log_threshold.requires_grad_(bool(train_threshold))
        self.register_buffer("initialized", torch.tensor(False))
        self.mem = None

    def threshold_shape(self):
        raise NotImplementedError

    def flatten_channels(self, x):
        raise NotImplementedError

    def threshold(self):
        return F.softplus(self.log_threshold) + self.eps

    def _preprocess(self, x):
        if self.signed:
            return x.abs(), torch.sign(x)
        return F.relu(x), None

    def _restore(self, x, sign):
        return x if sign is None else x * sign

    def _lazy_init_threshold(self, base):
        if self.initialized.item() or self.threshold_percentile <= 0:
            return
        with torch.no_grad():
            values = self.flatten_channels(base.detach().float())
            if values.numel() == 0:
                return
            quantile = torch.quantile(values, self.threshold_percentile, dim=0)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(quantile, op=dist.ReduceOp.SUM)
                quantile.div_(dist.get_world_size())
            quantile = quantile.clamp_min(self.eps).view(self.threshold_shape()).to(self.log_threshold.device)
            self.log_threshold.copy_(inverse_softplus(quantile))
            self.initialized.fill_(True)

    def reset(self):
        self.mem = None

    def forward(self, x):
        base, sign = self._preprocess(x)
        self._lazy_init_threshold(base)
        threshold = self.threshold().to(dtype=base.dtype, device=base.device)

        if self.mem is None or tuple(self.mem.shape) != tuple(base.shape):
            self.mem = torch.zeros_like(base) + 0.5 * threshold

        mem = self.mem + base
        hard = (mem >= threshold).to(base.dtype) * threshold
        scale = torch.clamp(threshold, min=self.eps)
        soft = torch.sigmoid(self.surrogate_alpha * (mem - threshold) / scale) * threshold
        spike = hard.detach() - soft.detach() + soft
        reset_value = hard.detach() if self.detach_reset else spike
        self.mem = (mem - reset_value).detach()
        return self._restore(spike, sign)


class TrainableSpike3dSeq(TrainableSpikeBase):
    def threshold_shape(self):
        return (1, 1, self.num_features)

    def flatten_channels(self, x):
        return x.reshape(-1, self.num_features)


class TrainableSpike5d(TrainableSpikeBase):
    def threshold_shape(self):
        return (1, self.num_features, 1, 1, 1)

    def flatten_channels(self, x):
        return x.permute(0, 2, 3, 4, 1).reshape(-1, self.num_features)


def import_multistep_lif_node():
    try:
        from spikingjelly.activation_based.neuron import MultiStepLIFNode
    except ImportError:
        from spikingjelly.clock_driven.neuron import MultiStepLIFNode
    return MultiStepLIFNode


class SpikingJellyLIFBase(nn.Module):
    def __init__(
        self,
        num_features,
        signed=True,
        tau=2.0,
        detach_reset=True,
        backend="torch",
    ):
        super().__init__()
        self.num_features = int(num_features)
        self.signed = bool(signed)
        self.tau = float(tau)
        self.detach_reset = bool(detach_reset)
        self.backend = str(backend)
        MultiStepLIFNode = import_multistep_lif_node()
        self.node = MultiStepLIFNode(
            tau=self.tau,
            detach_reset=self.detach_reset,
            backend=self.backend,
        )

    def _preprocess(self, x):
        if self.signed:
            return x.abs(), torch.sign(x)
        return F.relu(x), None

    def _restore(self, x, sign):
        return x if sign is None else x * sign

    def reset(self):
        if hasattr(self.node, "reset"):
            self.node.reset()
        elif hasattr(self.node, "v"):
            self.node.v = None


class SpikingJellyLIF3dSeq(SpikingJellyLIFBase):
    def forward(self, x):
        base, sign = self._preprocess(x)
        spike = self.node(base.unsqueeze(0)).squeeze(0)
        return self._restore(spike, sign)


class SpikingJellyLIF5d(SpikingJellyLIFBase):
    def forward(self, x):
        base, sign = self._preprocess(x)
        spike = self.node(base.unsqueeze(0)).squeeze(0)
        return self._restore(spike, sign)


def make_spike_module(kind, rank, num_features, spike_kwargs):
    kind = str(kind).lower()
    if kind == "trainable":
        trainable_kwargs = {
            key: spike_kwargs[key]
            for key in (
                "threshold_init",
                "threshold_percentile",
                "train_threshold",
                "signed",
                "surrogate_alpha",
                "detach_reset",
            )
        }
        if rank == "5d":
            return TrainableSpike5d(num_features, **trainable_kwargs)
        return TrainableSpike3dSeq(num_features, **trainable_kwargs)
    if kind == "lif":
        lif_kwargs = dict(
            signed=spike_kwargs["signed"],
            tau=spike_kwargs["lif_tau"],
            detach_reset=spike_kwargs["detach_reset"],
            backend=spike_kwargs["lif_backend"],
        )
        if rank == "5d":
            return SpikingJellyLIF5d(num_features, **lif_kwargs)
        return SpikingJellyLIF3dSeq(num_features, **lif_kwargs)
    raise ValueError(f"Unsupported spike_layer: {kind}")


class TrainableVideoMambaSNN(CleanVideoMamba):
    def __init__(
        self,
        *args,
        spike_patch=False,
        spike_block_indices=(0,),
        spike_position="post",
        spike_layer="trainable",
        snn_timesteps=4,
        signed_spikes=True,
        threshold_init=1.0,
        threshold_percentile=0.99,
        train_threshold=True,
        surrogate_alpha=4.0,
        lif_tau=2.0,
        lif_backend="torch",
        detach_reset=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.spike_patch = bool(spike_patch)
        self.spike_block_indices = tuple(sorted(set(int(x) for x in spike_block_indices)))
        self.spike_position = str(spike_position).lower()
        if self.spike_position not in {"pre", "post", "prepost"}:
            raise ValueError(f"Unsupported spike_position: {spike_position}")
        self.spike_layer = str(spike_layer).lower()
        if self.spike_layer not in {"trainable", "lif"}:
            raise ValueError(f"Unsupported spike_layer: {spike_layer}")
        self.snn_timesteps = int(snn_timesteps)
        self.signed_spikes = bool(signed_spikes)
        self.lif_tau = float(lif_tau)
        self.lif_backend = str(lif_backend)
        self.detach_reset = bool(detach_reset)

        spike_kwargs = dict(
            threshold_init=threshold_init,
            threshold_percentile=threshold_percentile,
            train_threshold=train_threshold,
            signed=signed_spikes,
            surrogate_alpha=surrogate_alpha,
            lif_tau=lif_tau,
            lif_backend=lif_backend,
            detach_reset=detach_reset,
        )
        self.patch_spike = make_spike_module(self.spike_layer, "5d", self.embed_dim, spike_kwargs)
        self.pre_block_spikes = nn.ModuleDict(
            OrderedDict(
                (str(idx), make_spike_module(self.spike_layer, "3d", self.embed_dim, spike_kwargs))
                for idx in self.spike_block_indices
                if self.spike_position in {"pre", "prepost"}
            )
        )
        self.block_spikes = nn.ModuleDict(
            OrderedDict(
                (str(idx), make_spike_module(self.spike_layer, "3d", self.embed_dim, spike_kwargs))
                for idx in self.spike_block_indices
                if self.spike_position in {"post", "prepost"}
            )
        )

    def iter_spike_modules(self):
        if self.spike_patch:
            yield self.patch_spike
        for idx in self.spike_block_indices:
            key = str(idx)
            if key in self.pre_block_spikes:
                yield self.pre_block_spikes[key]
        for idx in self.spike_block_indices:
            key = str(idx)
            if key in self.block_spikes:
                yield self.block_spikes[key]

    def active_spike_module_names(self):
        names = []
        if self.spike_patch:
            names.append("patch_spike")
        for idx in self.spike_block_indices:
            key = str(idx)
            if key in self.pre_block_spikes:
                names.append(f"pre_block_spikes.{idx}")
            if key in self.block_spikes:
                names.append(f"block_spikes.{idx}")
        return names

    def active_spike_parameter_names(self):
        if self.spike_layer != "trainable":
            return []
        names = []
        if self.spike_patch:
            names.append("patch_spike.log_threshold")
        for idx in self.spike_block_indices:
            key = str(idx)
            if key in self.pre_block_spikes:
                names.append(f"pre_block_spikes.{idx}.log_threshold")
            if key in self.block_spikes:
                names.append(f"block_spikes.{idx}.log_threshold")
        return names

    def reset_spike_state(self):
        for module in self.iter_spike_modules():
            module.reset()

    def forward_features(self, x, inference_params=None):
        x = self.patch_embed(x)
        if self.spike_patch:
            x = self.patch_spike(x)

        bsz, channels, timesteps, height, width = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(bsz * timesteps, height * width, channels)

        x = x + self.pos_embed[:, 1:, :]
        cls_tokens = self.cls_token.expand(bsz, -1, -1) + self.pos_embed[:, :1, :]
        x = rearrange(x, "(b t) n d -> (b n) t d", b=bsz, t=timesteps)
        x = x + self.temporal_pos_embedding
        x = rearrange(x, "(b n) t d -> b (t n) d", b=bsz, t=timesteps)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)

        residual = None
        hidden_states = x
        for idx, layer in enumerate(self.layers):
            key = str(idx)
            if key in self.pre_block_spikes:
                hidden_states = self.pre_block_spikes[key](hidden_states)

            if self.use_checkpoint and idx < self.checkpoint_num:
                hidden_states, residual = layer(
                    hidden_states,
                    residual,
                    inference_params=inference_params,
                    use_checkpoint=True,
                )
            else:
                hidden_states, residual = layer(
                    hidden_states,
                    residual,
                    inference_params=inference_params,
                )

            if key in self.block_spikes:
                hidden_states = self.block_spikes[key](hidden_states)

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        if self.use_mean_pooling:
            return hidden_states[:, 1:, :].mean(dim=1)
        return hidden_states[:, 0, :]

    def forward_single(self, video):
        steps = max(1, int(self.snn_timesteps))
        self.reset_spike_state()
        logits_sum = None
        for _ in range(steps):
            features = self.forward_features(video)
            logits = self.head(self.head_drop(features))
            logits_sum = logits if logits_sum is None else logits_sum + logits
        self.reset_spike_state()
        return logits_sum / float(steps)


def create_videomamba_small_trainable_snn(
    num_classes,
    img_size=224,
    num_frames=16,
    tubelet_size=1,
    drop_path=0.0,
    fc_drop_rate=0.0,
    use_mean_pooling=True,
    spike_patch=False,
    spike_block_indices=(0,),
    spike_position="post",
    spike_layer="trainable",
    snn_timesteps=4,
    signed_spikes=True,
    threshold_init=1.0,
    threshold_percentile=0.99,
    train_threshold=True,
    surrogate_alpha=4.0,
    lif_tau=2.0,
    lif_backend="torch",
    detach_reset=True,
):
    return TrainableVideoMambaSNN(
        img_size=img_size,
        patch_size=16,
        embed_dim=384,
        depth=24,
        rms_norm=False,
        residual_in_fp32=True,
        fused_add_norm=False,
        kernel_size=tubelet_size,
        num_frames=num_frames,
        num_classes=num_classes,
        drop_path_rate=drop_path,
        fc_drop_rate=fc_drop_rate,
        use_mean_pooling=use_mean_pooling,
        spike_patch=spike_patch,
        spike_block_indices=spike_block_indices,
        spike_position=spike_position,
        spike_layer=spike_layer,
        snn_timesteps=snn_timesteps,
        signed_spikes=signed_spikes,
        threshold_init=threshold_init,
        threshold_percentile=threshold_percentile,
        train_threshold=train_threshold,
        surrogate_alpha=surrogate_alpha,
        lif_tau=lif_tau,
        lif_backend=lif_backend,
        detach_reset=detach_reset,
    )
