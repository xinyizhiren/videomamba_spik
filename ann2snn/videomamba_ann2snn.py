from collections import OrderedDict

import torch
import torch.nn as nn
from einops import rearrange

from ann2snn.slayers import SpikingNeuron3dSeq, SpikingNeuron5d
from models.videomamba import RMSNorm, layer_norm_fn, rms_norm_fn
from models.videomamba_clean import CleanVideoMamba


class ConvertedVideoMambaSNN(CleanVideoMamba):
    def __init__(
        self,
        *args,
        spike_patch=True,
        spike_block_indices=(0, 1),
        signed_spikes=True,
        threshold_scale=1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.spike_patch = bool(spike_patch)
        self.spike_block_indices = tuple(sorted(set(int(x) for x in spike_block_indices)))
        self.signed_spikes = bool(signed_spikes)
        self.threshold_scale = float(threshold_scale)

        self.patch_spike = SpikingNeuron5d(
            self.embed_dim,
            c=self.threshold_scale,
            signed=self.signed_spikes,
        )
        self.block_spikes = nn.ModuleDict(
            OrderedDict(
                (
                    str(idx),
                    SpikingNeuron3dSeq(
                        self.embed_dim,
                        c=self.threshold_scale,
                        signed=self.signed_spikes,
                    ),
                )
                for idx in self.spike_block_indices
            )
        )

    def iter_spike_modules(self):
        if self.spike_patch:
            yield self.patch_spike
        for idx in self.spike_block_indices:
            yield self.block_spikes[str(idx)]

    def reset_spike_state(self):
        for module in self.iter_spike_modules():
            module.reset()

    def set_spike_mode(self, mode):
        for module in self.iter_spike_modules():
            module.mode = mode

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

            key = str(idx)
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


def create_videomamba_small_ann2snn(
    num_classes,
    img_size=224,
    num_frames=16,
    tubelet_size=1,
    drop_path=0.0,
    fc_drop_rate=0.0,
    use_mean_pooling=True,
    spike_patch=True,
    spike_block_indices=(0, 1),
    signed_spikes=True,
    threshold_scale=1.0,
):
    return ConvertedVideoMambaSNN(
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
        signed_spikes=signed_spikes,
        threshold_scale=threshold_scale,
    )
