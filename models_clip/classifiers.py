import torch
import torch.nn as nn
import torch.nn.functional as F


class _Classifier(nn.Module):
    def __init__(self, feat_dim=None, num_classes=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, feat_dim, dtype=dtype))
        # self.weight.data.uniform_(-1, 1).renorm_(2, 0, 1e-5).mul_(1e5)
        self.weight.data.uniform_(-1, 1).renorm_(2, 0, 1.0).mul_(1.0)

        # print("weight", self.weight)
        # exit(0)

    @property
    def dtype(self):
        return self.weight.dtype

    def forward(self, x):
        raise NotImplementedError

    def apply_weight(self, weight):
        self.weight.data = weight.clone()
    

class LinearClassifier(_Classifier):
    def __init__(self, feat_dim=None, num_classes=None, dtype=None, **kwargs):
        super().__init__(feat_dim, num_classes, dtype)
        nn.init.kaiming_normal_(self.weight.data)
        self.bias = nn.Parameter(torch.zeros(num_classes, dtype=dtype))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class CosineClassifier(nn.Module):
    """Cosine classifier with the matmul exposed as a real nn.Linear submodule.

    Why the structure: the previous implementation stored the classifier
    weight as a raw nn.Parameter and called F.linear in forward. That made
    the head invisible to module-level tooling (replace_linears, GPTQ's
    forward_pre_hook), so post-training quantization silently skipped it.
    By holding the matmul on `self.linear`, the head participates in the
    same QuantLinear swap path as the backbone, and `--scope full`
    actually quantizes it.

    Note: the legacy forward applied F.normalize to the weight on every
    call. After init_clip_head_from_text writes already-L2-normalized text
    features into head.linear.weight that step is a near-identity on the
    FP32 path, so dropping it preserves training-time behavior. After
    quantization, weight rows are no longer exactly unit-norm — the same
    trade-off SAP-v2's `CosineClassifier(nn.Linear)` makes.
    """

    def __init__(self, feat_dim=None, num_classes=None, dtype=None, scale=25, **kwargs):
        super().__init__()
        self.linear = nn.Linear(feat_dim, num_classes, bias=False, dtype=dtype)
        self.linear.weight.data.uniform_(-1, 1).renorm_(2, 0, 1.0).mul_(1.0)
        self.scale = scale

    @property
    def weight(self):
        # Backwards compatibility for code that reads head.weight.
        return self.linear.weight

    @property
    def dtype(self):
        return self.linear.weight.dtype

    def apply_weight(self, weight):
        self.linear.weight.data = weight.clone()

    def forward(self, x):
        x = F.normalize(x, dim=-1)
        return self.linear(x) * self.scale


class L2NormedClassifier(_Classifier):
    def __init__(self, feat_dim=None, num_classes=None, dtype=None, **kwargs):
        super().__init__(feat_dim, num_classes, dtype)
    
    def forward(self, x):
        weight = F.normalize(self.weight, dim=-1)
        return F.linear(x, weight)


class LayerNormedClassifier(_Classifier):
    def __init__(self, feat_dim=None, num_classes=None, dtype=None, **kwargs):
        super().__init__(feat_dim, num_classes, dtype)
        self.ln = nn.LayerNorm(feat_dim, elementwise_affine=False, eps=1e-12, dtype=dtype)

    def forward(self, x):
        x = self.ln(x)
        weight = F.normalize(self.weight, dim=-1)
        return F.linear(x, weight)
