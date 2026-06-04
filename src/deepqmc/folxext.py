"""Wide-scope folx-compatible forward-Laplacian rule for multi-head attention.

Mirrors lapnet/networks/transformer_blocks.py::attention_sparse_dot_product (which
uses lapjax LapTuples) into folx's FwdLaplArray world.

Function exposed to folx:

    sparse_attention(q, k, v)  ==  softmax(q @ k.T / sqrt(D_k)) @ v

with q, k, v of shape (..., H, N, D_per_head).  Registered via
folx.register_function so a plain `ForwardLaplacianOperator(thr)(fn)` routes to
the rule whenever `sparse_attention` appears in the traced graph.

Why wide-scope?  The matmul `exp(logits) @ v` is done while exp(logits)'s
Jacobian is still in its sparse 2k_x-slot form (i-side + j-side), and `sum_j
exp(logits)` is collapsed in this same sparse basis.  Only the final divide
`/ sum_exp` densifies the output Jacobian.  Folx's default would densify the
logits Jacobian very early (at the QK^T step), spending O(n_inputs · N² · D_k)
on something a hand-written rule does in O(k_x · N² · D_k) ~ k_x/n_inputs ≈
1/N cheaper.

Assumptions about inputs (matching LapNet's setting):
  - q, k come from per-electron Linear projections of the same upstream x of
    shape (N, k_x), so q_arr.jacobian.x0_idx[s, ..., i, e] = i*k_x + s
    (independent of head h and embedding e).
  - v likewise has the same sparsity structure (weak).  If v is dense
    (e.g. someone normalized it outside this rule), the rule still works but
    pays the full n_inputs · N² · D_v matmul cost — the same place we got
    stuck in the kqT_v rule.
"""
import folx
from folx import register_function
from folx.api import FwdJacobian, FwdLaplArray
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


# ============================================================================
# 1) Pure-jax forward (the thing folx routes to the custom rule)
# ============================================================================

@jax.jit
def sparse_attention(q, k, v):
    """Scaled dot product attention with softmax.

    q, k, v: (..., H, N, D_per_head).
    Returns:  (..., H, N, D_per_head).
    """
    d_k = q.shape[-1]
    logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) / jnp.sqrt(d_k)
    attn = jax.nn.softmax(logits, axis=-1)
    return jnp.matmul(attn, v)


# ============================================================================
# 2) Custom forward-Laplacian rule (closely follows LapNet's algorithm)
# ============================================================================

def _sparse_attention_rule(args, kwargs, sparsity_threshold):
    print("Using Claude's original implementation.")
    """Wide-scope forward Laplacian for `sparse_attention(q, k, v)`."""
    q_arr, k_arr, v_arr = args
    # print(q_arr.shape)
    # print(q_arr.laplacian.shape)
    # print(q_arr.jacobian.weak)
    # print(q_arr.jacobian.dense_array.shape)
    # print(q_arr.jacobian.data.shape)
    # print(q_arr.jacobian)
    assert q_arr.jacobian.weak, "expected weak q (Linear projection of x)"
    assert k_arr.jacobian.weak, "expected weak k"

    q, k, v = q_arr.x, k_arr.x, v_arr.x
    H, N, D_k = q.shape
    D_v = v.shape[-1]
    scale = 1.0 / jnp.sqrt(D_k)

    dq, dk = q_arr.jacobian.data, k_arr.jacobian.data       # (k_x, H, N, D_*)
    k_x = dq.shape[0]
    lap_q, lap_k = q_arr.laplacian, k_arr.laplacian
    lap_v = v_arr.laplacian

    # Densify v.jacobian here (LapNet does the same — `v = v.set_dense(force=True)`)
    if v_arr.jacobian.weak:
        v_jac = v_arr.jacobian.dense_array
    else:
        v_jac = v_arr.jacobian.data                          # (n_inputs, H, N, D_v)
    n_inputs = v_jac.shape[0]
    assert n_inputs == N * k_x, "expecting n_inputs = N · k_x for the Linear-of-x setting"
    # Re-shape so the n_inputs axis splits into (electron, slot)
    v_jac_r = v_jac.reshape(N, k_x, H, N, D_v)               # (p, s, h, t, d)

    eye_N = jnp.eye(N, dtype=q.dtype)
    arange_N = jnp.arange(N)

    # ----------------------------------------------------------------------
    # Step 1:  logits = q @ k.T · scale         (weak 2k_x-slot Jacobian)
    # ----------------------------------------------------------------------
    logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) * scale

    # dlogits split into i-side (from dq) and j-side (from dk).  Diag(i==j)
    # contributions of j-side moved into i-side so the two halves carry
    # disjoint x0_idx ranges (lapnet does the same dance).
    dlogits_i = jnp.einsum('shid,hjd->shij', dq, k) * scale  # (k_x, H, N, N)
    dlogits_j = jnp.einsum('hid,shjd->shij', q, dk) * scale  # (k_x, H, N, N)
    diag_j = dlogits_j[..., arange_N, arange_N]
    dlogits_i = dlogits_i.at[..., arange_N, arange_N].add(diag_j)
    dlogits_j = dlogits_j.at[..., arange_N, arange_N].add(-diag_j)

    # lap_logits = lap(q)@k.T + q@lap(k).T + 2·cross_kq on diagonal
    cross_kq_diag = jnp.einsum('shid,shid->hi', dq, dk)      # (H, N)
    lap_logits = (
        jnp.matmul(lap_q, jnp.swapaxes(k, -2, -1)) * scale
        + jnp.matmul(q, jnp.swapaxes(lap_k, -2, -1)) * scale
        + 2 * cross_kq_diag[..., None] * eye_N * scale
    )

    # ----------------------------------------------------------------------
    # Step 2:  exp(logits)             (chain rule, still weak)
    # ----------------------------------------------------------------------
    max_logits = jax.lax.stop_gradient(jnp.max(logits, axis=-1, keepdims=True))
    exp_logits = jnp.exp(logits - max_logits)                # (H, N, N)
    dexp_i = exp_logits[None] * dlogits_i                    # (k_x, H, N, N)
    dexp_j = exp_logits[None] * dlogits_j
    # |grad logits|² for the exp chain-rule.  The i-side and j-side x0_idx
    # families are disjoint, so the squares add.
    grad_logits_sq = (dlogits_i ** 2).sum(0) + (dlogits_j ** 2).sum(0)
    lap_exp = exp_logits * (lap_logits + grad_logits_sq)

    # ----------------------------------------------------------------------
    # Step 3:  sum_exp = sum_j exp_logits  +  Y_num = exp_logits @ v
    # ----------------------------------------------------------------------
    sum_exp = jnp.sum(exp_logits, axis=-1, keepdims=True)    # (H, N, 1)
    sum_exp_v = jnp.matmul(exp_logits, v)                    # (H, N, D_v)

    # -- gradient of sum_exp (dense over n_inputs) --
    # i-side stays weak (sum_j doesn't touch the i-indexed slots):
    dsum_exp_i_weak = dexp_i.sum(-1)                          # (k_x, H, N)
    # j-side densifies (each j becomes its own input slot):
    # dsum_exp[a=j*k_x+s, h, i] = dexp_j[s, h, i, j]
    dsum_exp_j_dense = dexp_j.transpose(3, 0, 1, 2)           # (N=j, k_x, H, N=i)
    dsum_exp_j_dense = dsum_exp_j_dense.reshape(N * k_x, H, N)
    # i-side scattered to dense form:
    dsum_exp_i_dense = (
        dsum_exp_i_weak.transpose(2, 0, 1)[:, :, :, None]    # (N=i_pos, k_x, H, 1)
        * eye_N[:, None, None, :]                             # (N=i_pos, 1, 1, N=i_jac)
    ).reshape(N * k_x, H, N)
    dsum_exp = dsum_exp_i_dense + dsum_exp_j_dense           # (n_inputs, H, N)

    lap_sum_exp = lap_exp.sum(-1)                             # (H, N)

    # -- gradient of sum_exp_v = exp_logits @ v  (dense over n_inputs) --
    # 3a) exp_logits @ dv   — the irreducible matmul, structured via the
    #     (p, s, h, t, d) reshape of v_jac.
    sum_exp_v_grad_v = jnp.einsum('hit,pshtd->pshid', exp_logits, v_jac_r)  # (N, k_x, H, N, D_v)
    # 3b) dexp_i  @ v       — sparse, placed at i-diagonal.
    sum_exp_v_grad_i = jnp.einsum('shij,hjd->shid', dexp_i, v)             # (k_x, H, N=i, D_v)
    sum_exp_v_grad_i_dense = (
        sum_exp_v_grad_i.transpose(2, 0, 1, 3)[:, :, :, None, :]            # (N=p, k_x, H, 1, D_v)
        * eye_N[:, None, None, :, None]                                      # (N=p, 1, 1, N=i_jac, 1)
    )                                                                        # (N, k_x, H, N, D_v)
    # 3c) dexp_j  @ v       — sparse, placed at j-row of n_inputs.
    sum_exp_v_grad_j = jnp.einsum('shij,hjd->jshid', dexp_j, v)            # (N=j, k_x, H, N, D_v)
    dsum_exp_v_r = sum_exp_v_grad_v + sum_exp_v_grad_i_dense + sum_exp_v_grad_j
    dsum_exp_v = dsum_exp_v_r.reshape(N * k_x, H, N, D_v)

    # -- Laplacian of sum_exp and sum_exp_v --
    # lap(sum_exp_v) = lap(exp_logits) @ v + 2·cross(d_exp, d_v) + exp_logits @ lap(v)
    lap_sum_exp_v_a = jnp.matmul(lap_exp, v)                                # (H, N, D_v)
    lap_sum_exp_v_c = jnp.matmul(exp_logits, lap_v)
    # cross_i[h, i, d] = sum_{s, j} d_exp_i[s, h, i, j] · v_jac_r[i, s, h, j, d]
    #   (i-side weak: x0_idx = i*k_x + s, so the contributing v_jac row is i)
    cross_i = jnp.einsum('shij,ishjd->hid', dexp_i, v_jac_r)
    # cross_j[h, i, d] = sum_{s, j} d_exp_j[s, h, i, j] · v_jac_r[j, s, h, j, d]
    #   (j-side weak: x0_idx = j*k_x + s, contributing row of v_jac is j)
    v_jac_r_diag_jj = v_jac_r[arange_N, :, :, arange_N, :]                  # (N=j, k_x, H, D_v)
    cross_j = jnp.einsum('shij,jshd->hid', dexp_j, v_jac_r_diag_jj)
    lap_sum_exp_v = lap_sum_exp_v_a + 2 * (cross_i + cross_j) + lap_sum_exp_v_c

    # ----------------------------------------------------------------------
    # Step 4:  Y = sum_exp_v / sum_exp[..., None]    (final divide, densifies)
    # ----------------------------------------------------------------------
    Y = sum_exp_v / sum_exp                                                  # (H, N, D_v)
    # dY = (d sum_exp_v - Y · d sum_exp[..., None]) / sum_exp
    dY = (
        dsum_exp_v - Y[None] * dsum_exp[..., None]                           # (n_inputs, H, N, D_v)
    ) / sum_exp

    # lap(Y) for y = u/v:
    #   lap(y) = ( lap(u) − 2 ⟨du, dv⟩/v − y·lap(v) + 2·y·|dv|²/v ) / v
    # Here u = sum_exp_v (shape (H, N, D_v)), v_d = sum_exp (shape (H, N, 1)).
    dudv = (dsum_exp_v * dsum_exp[..., None]).sum(0)                         # ⟨du, dv⟩, (H, N, D_v)
    dv_sq = (dsum_exp ** 2).sum(0)                                            # |dv|², (H, N)
    lap_Y = (
        lap_sum_exp_v
        - 2 * dudv / sum_exp
        - Y * lap_sum_exp[..., None]
        + 2 * Y * dv_sq[..., None] / sum_exp
    ) / sum_exp

    return FwdLaplArray(Y, FwdJacobian.from_dense(dY), lap_Y)


register_function("sparse_attention", _sparse_attention_rule)
