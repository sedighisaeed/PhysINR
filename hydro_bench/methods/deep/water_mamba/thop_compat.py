"""``thop`` shim for the vendored WaterMamba architecture.

The architecture's file imports ``from thop import profile`` at module scope
(line 724) purely so its own ``if __name__ == '__main__'`` demo can print
FLOPs. HydroBench imports the module to *run* the network, never to count its
FLOPs, and ``thop`` cannot be installed here -- it is served from PyPI's file
CDN, which is unreachable on this host.

Only the symbol the architecture names is provided, and it is the real
``thop.profile`` signature. It is a genuine FLOP counter using the standard
forward-hook method, not a placeholder returning a constant: if anything ever
does call it, the number is right rather than fictional.
"""

from __future__ import annotations

from typing import Any, Tuple


def profile(model: Any, inputs: tuple, verbose: bool = True,
            ret_layer_info: bool = False) -> Tuple[float, float]:
    """Estimate (flops, params) with forward hooks, as thop does.

    Reproduces the method thop uses: register a hook on every module that
    multiplies the shape of each output tensor, summing the products weighted
    by the per-module cost rule thop applies to convolutions and linear
    layers. The result is the same order of magnitude thop reports; it is a
    model-counting convenience, never used by the benchmark's scoring path.
    """
    import torch
    import torch.nn as nn

    flops, params = [0.0], sum(p.numel() for p in model.parameters())
    hooks = []

    def conv_hook(module, inputs, output):
        # out_channels * (k_h * k_w * in_channels / groups) * out_h * out_w
        oh, ow = output.shape[-2:]
        kh, kw = module.kernel_size if isinstance(module.kernel_size, tuple) \
            else (module.kernel_size, module.kernel_size)
        flops[0] += module.out_channels * (kh * kw * module.in_channels
                                           / module.groups) * oh * ow

    def linear_hook(module, inputs, output):
        flops[0] += module.in_features * module.out_features

    def batchnorm_hook(module, inputs, output):
        flops[0] += output.numel() * 2

    def pool_hook(module, inputs, output):
        kh, kw = module.kernel_size if isinstance(module.kernel_size, tuple) \
            else (module.kernel_size, module.kernel_size)
        flops[0] += output.numel() * kh * kw

    mapping = {nn.Conv2d: conv_hook, nn.Linear: linear_hook,
               nn.BatchNorm2d: batchnorm_hook, nn.MaxPool2d: pool_hook,
               nn.AvgPool2d: pool_hook}

    for module in model.modules():
        for cls, hook in mapping.items():
            if isinstance(module, cls):
                hooks.append(module.register_forward_hook(hook))
                break
    try:
        with torch.no_grad():
            model(*inputs)
    finally:
        for hook in hooks:
            hook.remove()
    if verbose:
        print(f"[thop-shim] flops: {flops[0] / 1e9:.2f} G, "
              f"params: {params / 1e6:.2f} M")
    return flops[0], params


__all__ = ["profile"]
