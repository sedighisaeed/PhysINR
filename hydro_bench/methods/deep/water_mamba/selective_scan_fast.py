"""Memory-bounded selective scan for hosts without the CUDA kernel.

``selective_scan_ref`` above is the ``mamba_ssm`` authors' own pure-PyTorch
implementation, and it is what the CUDA kernel is validated against. But it
materialises ``deltaA`` and ``deltaB_u`` as full ``(B, D, L, N)`` tensors and
then steps the recurrence in a Python loop over ``L``. For a 0.36 Mpixel
image that is a 4 GiB allocation and ~100 s of Python per scan call -- across
the network's sixteen scans it would make one image take the better part of
an hour, and on Apple's GPU the allocation does not even fit.

The CUDA kernel avoids both by streaming the recurrence. This module reaches
the same arithmetic without CUDA by *chunking* the sequence and running a
parallel scan (Hillis-Steele) within each chunk:

* the sequence is cut into blocks of ``CHUNK`` timesteps, so no intermediate
  ever exceeds ``(B, D, CHUNK, N)``;
* inside a chunk the linear recurrence ``x_t = a_t x_{t-1} + b_t`` is
  resolved with ``log2(CHUNK)`` vectorised combine steps rather than a Python
  loop;
* the chunk's final state is carried into the next chunk, so the result is
  the same recurrence evaluated end to end.

The combination is associative -- ``(a, b) * (a', b') = (a'*a, a'*b + b')`` is
exactly "apply this linear map, then that one" -- so the parallel scan
computes the same quantity as the loop, to floating-point associativity. The
CUDA kernel is likewise not bit-identical to the reference; results here agree
with the reference to the tolerance asserted in the tests.

Only the forward pass is provided: HydroBench scores models, it never trains
them.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# The scan's index juggling is written in einops notation. The vendored
# architecture needs einops too, so it is a hard dependency of this path;
# importing it here surfaces a clear error rather than a NameError later.
from einops import rearrange, repeat

# Chunk length for the parallel scan. Bounded so that (B, D, CHUNK, N) stays
# small (a few tens of MB at this network's widths) while giving the
# vectorised combine steps long enough vectors to amortise their overhead.
CHUNK = 1024


def _parallel_scan(a: torch.Tensor, b: torch.Tensor) -> tuple:
    """Inclusive Hillis-Steele scan of ``x_t = a_t x_{t-1} + b_t``.

    ``a``, ``b``: (..., L, ...) with the sequence on dim ``seq_dim``. Returns
    ``(A, B)`` where the composition of steps ``0..t`` is ``x -> A_t x + B_t``,
    i.e. ``x_t = A_t x_start + B_t``.
    """
    seq_dim = -2
    length = a.shape[seq_dim]
    a, b = a.clone(), b.clone()
    stride = 1
    while stride < length:
        # position i takes the composition of the block ending at i with the
        # block of the same length before it: (a', b') applied after (a, b).
        a_prev = a[..., :-stride, :]
        b_prev = b[..., :-stride, :]
        a_cur = a[..., stride:, :]
        b_cur = b[..., stride:, :]
        new_a = a_cur * a_prev
        new_b = b_cur + a_cur * b_prev
        a[..., stride:, :] = new_a
        b[..., stride:, :] = new_b
        stride *= 2
    return a, b


def selective_scan_chunked(u, delta, A, B, C, D=None, z=None,
                           delta_bias=None, delta_softplus=False,
                           return_last_state=False, chunk: int = CHUNK):
    """Vectorised, memory-bounded selective scan; see module docstring.

    Signature and tensor conventions match ``selective_scan_ref`` exactly.
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    if A.is_complex():
        raise NotImplementedError(
            "the chunked scan supports real A; WaterMamba does not use complex "
            "state. Use selective_scan_ref for the complex path."
        )
    B = B.float()
    C = C.float()

    total = u.shape[2]
    # Chunk boundaries, aligned so every chunk is a full scan block except the
    # last (the parallel scan handles any length; alignment only keeps it even).
    edges = list(range(0, total, chunk)) or [0]

    x = A.new_zeros((batch, dim, dstate))
    ys = []
    last_state = None
    for start in edges:
        end = min(start + chunk, total)
        u_c = u[:, :, start:end]
        d_c = delta[:, :, start:end]
        # (B, D, C, N): the only large allocation, bounded by the chunk.
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', d_c, A))
        if not is_variable_B:
            deltaB_u = torch.einsum('bdl,dn,bdl->bdln', d_c, B, u_c)
        elif B.dim() == 3:
            deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', d_c,
                                    B[:, :, start:end], u_c)
        else:
            B_c = B[:, :, :, start:end]
            if B.shape[1] != dim:
                B_c = repeat(B_c, "B G N L -> B (G H) N L", H=dim // B.shape[1])
            deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', d_c, B_c, u_c)

        A_t, B_t = _parallel_scan(deltaA, deltaB_u)
        # Fold the carried state in, then evaluate the output at each step.
        x_t = A_t * x.unsqueeze(2) + B_t          # (B, D, C, N)

        if not is_variable_C:
            # C: (D, N) -- constant across the sequence.
            y = torch.einsum('bdcn,dn->bdc', x_t, C)
        elif C.dim() == 3:
            # C: (B, N, L) -- per-step, shared across the D channels.
            y = torch.einsum('bdcn,bnc->bdc', x_t, C[:, :, start:end])
        else:
            # C: (B, G, N, L) with G folded into D -- per-step and per-channel.
            C_c = C[:, :, :, start:end]
            if C.shape[1] != dim:
                C_c = repeat(C_c, "B G N L -> B (G H) N L", H=dim // C.shape[1])
            y = torch.einsum('bdcn,bdnc->bdc', x_t, C_c)
        ys.append(y)

        x = x_t[:, :, -1, :]
        if end == total:
            last_state = x

    y = torch.cat(ys, dim=2)                      # (B, D, L)
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)
    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, last_state)
