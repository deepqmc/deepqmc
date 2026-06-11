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

import jax
import jax.numpy as jnp
from folx import register_function
from folx.api import FwdJacobian, FwdLaplArray


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


def _sparse_attention_rule(args, kwargs, sparsity_threshold):
    q, k, v = args
    q_value, k_value, v_value = q.x, k.x, v.x
    q_jacobian, k_jacobian, v_jacobian = q.jacobian.dense_array, k.jacobian.dense_array, v.jacobian.dense_array
    q_lap, k_lap, v_lap = q.laplacian, k.laplacian, v.laplacian
    
    N = q_value.shape[-2]
    d_k = q_value.shape[-1]
    F = v_value.shape[-1]
    D = q_jacobian.shape[0] // N
    batch_shape = q_value.shape[:-2]
    scale = 1.0 / jnp.sqrt(d_k)
    
    # 1. Forward values (Deferred)
    S = jnp.matmul(q_value, jnp.swapaxes(k_value, -2, -1)) * scale
    S_max = jax.lax.stop_gradient(jnp.max(S, axis=-1, keepdims=True))
    E = jnp.exp(S - S_max) # (..., N, N)
    Z = jnp.sum(E, axis=-1) # (..., N)
    U = jnp.matmul(E, v_value) # (..., N, F)
    h = U / jnp.expand_dims(Z, -1) # (..., N, F)
    
    # Block-sparse diagonals extraction via ellipsis matching
    q_jac_unflat = q_jacobian.reshape(N, D, *batch_shape, N, d_k)
    k_jac_unflat = k_jacobian.reshape(N, D, *batch_shape, N, d_k)
    dQ = jnp.einsum('n d ... n f -> ... n d f', q_jac_unflat) # (..., N_out, D, d_k)
    dK = jnp.einsum('n d ... n f -> ... n d f', k_jac_unflat) # (..., N_out, D, d_k)
    
    # Precompute common O(N^2 D) cross-particle projections
    QdK_cross = jnp.einsum('...ia, ...mda -> ...imd', q_value, dK) # (..., N_out, N_deriv, D)
    QdK = jnp.swapaxes(QdK_cross, -1, -2) # (..., N_out, D, N_deriv)
    
    # Identity matrix for scattering diagonals
    I = jnp.eye(N, dtype=q_value.dtype)
    
    # 2. Jacobian of Z (Denominator)
    EK = jnp.matmul(E, k_value) # (..., N, d_k)
    t1_Z = jnp.einsum('...idf, ...if -> i d ...', dQ, EK) * scale
    dZ_t1_full = jnp.einsum('id..., ij -> i d ... j', t1_Z, I) # Places t1_Z on the i=j diagonal
    
    t2_Z = jnp.einsum('...im, ...imd -> m d ... i', E, QdK_cross) * scale
    
    dZ_unflat = dZ_t1_full + t2_Z # (N, D, ..., N)
    dZ = dZ_unflat.reshape(D * N, *batch_shape, N) # (C, ..., N)
    
    # 3. Jacobian of U (Unnormalized Output)
    KV = jnp.einsum('...ja, ...jf -> ...jaf', k_value, v_value)
    EKV = jnp.einsum('...ij, ...jaf -> ...iaf', E, KV) # (..., N, d_k, F)
    
    t1a_U = jnp.einsum('...ida, ...iaf -> i d ... f', dQ, EKV) * scale
    t1a_U_full = jnp.einsum('id...f, ij -> i d ... j f', t1a_U, I) # Places t1a_U on i=j diagonal
    
    t1b_U = jnp.einsum('...im, ...imd, ...mf -> m d ... i f', E, QdK_cross, v_value) * scale
    
    dU_unflat = t1a_U_full + t1b_U
    
    # Broadcast E across derivative space to matmul with v_jacobian directly
    E_expanded = jnp.expand_dims(E, 0)
    dU = dU_unflat.reshape(D * N, *batch_shape, N, F) + jnp.matmul(E_expanded, v_jacobian)
    
    # 4. Jacobian of Output (h) via Quotient Rule
    dh = (dU - jnp.expand_dims(h, 0) * jnp.expand_dims(dZ, -1)) / jnp.expand_dims(Z, (0, -1))
    
    # 5. Laplacian of E and Z
    cross_S_diag_val = jnp.einsum('...idf, ...idf -> ...i', dQ, dK) * 2 * scale
    lap_S_cross = jnp.einsum('...i, ij -> ...ij', cross_S_diag_val, I)
    
    lap_S = (
        jnp.einsum('...ia, ...ja -> ...ij', q_lap, k_value) * scale + 
        jnp.einsum('...ia, ...ja -> ...ij', q_value, k_lap) * scale + 
        lap_S_cross
    )
    
    dQK = jnp.einsum('...ida, ...ja -> ...idj', dQ, k_value)
    
    dQK_diag = jnp.einsum('...ida, ...ia -> ...id', dQ, k_value)
    QdK_diag = jnp.einsum('...ia, ...ida -> ...id', q_value, dK)
    
    norm_S = (jnp.sum(dQK**2, axis=-2) + jnp.sum(QdK**2, axis=-2)) * (scale**2)
    cross_S_diag_val2 = 2 * jnp.sum(dQK_diag * QdK_diag, axis=-1) * (scale**2)
    norm_S = norm_S + jnp.einsum('...i, ij -> ...ij', cross_S_diag_val2, I)
    
    lap_E = E * (lap_S + norm_S)
    lap_Z = jnp.sum(lap_E, axis=-1)
    
    # 6. Laplacian of U
    dV_unflat = v_jacobian.reshape(N, D, *batch_shape, N, F)
    dV_diag = jnp.einsum('m d ... m f -> ... m d f', dV_unflat)
    
    E_dQK = jnp.expand_dims(E, -2) * dQK # (..., i, d, j)
    t2a_U_lap = jnp.einsum('...idj, i d ... j f -> ...if', E_dQK, dV_unflat) * scale
    
    E_QdK = jnp.expand_dims(E, -2) * QdK # (..., i, d, j)
    t2b_U_lap = jnp.einsum('...idj, ...jdf -> ...if', E_QdK, dV_diag) * scale
    
    lap_U = (
        jnp.matmul(lap_E, v_value) + 
        jnp.matmul(E, v_lap) + 
        2 * (t2a_U_lap + t2b_U_lap)
    )
    
    # 7. Laplacian of Output (h) via Quotient Rule
    dh_dot_dZ = jnp.sum(dh * jnp.expand_dims(dZ, -1), axis=0) # (..., N, F)
    lap_h = (lap_U - 2 * dh_dot_dZ - h * jnp.expand_dims(lap_Z, -1)) / jnp.expand_dims(Z, -1)
    
    return FwdLaplArray(h, FwdJacobian.from_dense(dh), lap_h)

register_function('sparse_attention', _sparse_attention_rule)
