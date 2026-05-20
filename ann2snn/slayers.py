import torch
import torch.nn as nn
import torch.nn.functional as F


class Dummy(nn.Module):
    def forward(self, x):
        return x


class _BaseSpikingNeuron(nn.Module):
    def __init__(self, c=1.0, mode="ann", signed=True):
        super().__init__()
        self.mode = mode
        self.signed = bool(signed)
        self.register_buffer("c", torch.as_tensor(c, dtype=torch.float32))
        self.T = 0
        self.mem = 0.0
        self.delta = 0.0
        self.r = 0.0

    def _preprocess(self, x):
        if self.signed:
            sign = torch.sign(x)
            base = x.abs()
        else:
            sign = None
            base = F.relu(x)
        return base, sign

    def _restore(self, x, sign):
        return x if sign is None else x * sign

    def _reduce_dims(self, x):
        raise NotImplementedError

    def _get_threshold(self, x):
        return self.thre

    def _clip_to_threshold(self, x, threshold):
        return torch.minimum(torch.clamp_min(x, 0.0), threshold)

    def optimize(self, x):
        base, sign = self._preprocess(x)
        ub = self._get_threshold(base)
        reduce_dims = self._reduce_dims(base)
        self.thre += self.c * (2 * (base - ub) * (base > ub)).mean(reduce_dims, keepdim=True)
        self.delta = ((base - ub) * (base > ub).float()).mean().item()

        clipped = self._clip_to_threshold(base, ub)
        return self._restore(clipped, sign)

    def forward(self, x):
        if self.mode == "passthrough":
            return x

        base, sign = self._preprocess(x)
        ub = self._get_threshold(base)
        reduce_dims = self._reduce_dims(base)

        if self.mode == "snn":
            if self.T == 0:
                self.mem = 0.5 * ub
            self.mem = self.mem + base
            spike = (self.mem > ub).float() * ub
            self.mem = self.mem - spike
            self.T += 1
            return self._restore(spike, sign)

        if self.mode == "clip":
            denom = base.mean(reduce_dims, keepdim=False).max() + 1e-6
            self.r = (ub.mean() / 2.0 / denom).item()
            clipped = self._clip_to_threshold(base, ub)
            return self._restore(clipped, sign)

        if self.mode == "robust_norm":
            q = base.reshape(-1).detach().quantile(0.99, interpolation="nearest")
            self.thre = torch.maximum(self.thre, q.expand_as(self.thre))
            return self._restore(base, sign)

        return self.optimize(x)

    def reset(self):
        self.mem = 0.0
        self.T = 0
        self.delta = 0.0
        self.r = 0.0


class SpikingNeuron2d(_BaseSpikingNeuron):
    def __init__(self, num_features, c=1.0, mode="ann", signed=True):
        super().__init__(c=c, mode=mode, signed=signed)
        self.register_buffer("thre", torch.zeros((1, num_features)))
        self.num_features = num_features

    def _reduce_dims(self, x):
        return (0,)


class SpikingNeuron3dSeq(_BaseSpikingNeuron):
    def __init__(self, num_features, c=1.0, mode="ann", signed=True):
        super().__init__(c=c, mode=mode, signed=signed)
        self.register_buffer("thre", torch.zeros((1, 1, num_features)))
        self.num_features = num_features

    def _reduce_dims(self, x):
        return (0, 1)


class SpikingNeuron5d(_BaseSpikingNeuron):
    def __init__(self, num_features, c=1.0, mode="ann", signed=True):
        super().__init__(c=c, mode=mode, signed=signed)
        self.register_buffer("thre", torch.zeros((1, num_features, 1, 1, 1)))
        self.num_features = num_features

    def _reduce_dims(self, x):
        return (0, 2, 3, 4)
