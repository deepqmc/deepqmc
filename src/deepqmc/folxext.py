import jax
import jax.numpy as jnp
from folx import register_function
from folx.api import FwdJacobian, FwdLaplArray
from folx.interpreter import forward_laplacian


def _mulsum(S, x):
    return (S[..., :, :, None] * x[..., None, :, :]).sum(-2)


def _asymmetric_lap_rule(S: FwdLaplArray, v: FwdLaplArray):
    """Forward-Laplacian for Y = S @ v with weak S and dense v.

    Supports arbitrary leading batch axes:
      S.x:   (*batch, T', T)
      v.x:   (*batch, T,  V)
      S.jac: (k_S,     *batch, T', T) (weak)
      v.jac: (n_inputs, *batch, T,  V) (dense)
      out:   (*batch, T', V)
      dY:    (n_inputs, *batch, T', V)

    dY     = dS @ v   +   S @ dv
             (sparse)      (dense, unavoidable n_inputs matmuls)
    lap Y  = lap_S @ v  +  2·cross(dS, dv)  +  S @ lap_v
             (cross exploits dS sparsity)
    """
    assert S.jacobian.weak and not v.jacobian.weak
    n_inputs = v.jacobian.data.shape[0]
    n_batch = S.x.ndim - 2

    # primal
    Y_x = jnp.matmul(S.x, v.x)

    # ---- jacobian: dY = dS @ v + S @ dv ----
    # S @ dv: '...ij, a...jd -> a...id'
    S_dv = jnp.einsum('...ij,a...jd->a...id', S.x, v.jacobian.data)

    # dS @ v: contrib[k, ..., i, j, d] = dS.data[k, ..., i, j] · v.x[..., j, d]
    contrib = jnp.einsum('k...ij,...jd->k...ijd', S.jacobian.data, v.x)
    x0_idx = jnp.asarray(S.jacobian.x0_idx)  # (k_S, *batch, T', T)

    # Move k_S inward so (T', batch...) end up as leading vmap axes.
    contrib_r = jnp.moveaxis(contrib, 0, -3)  # (*batch, T', k_S, T, D)
    idx_r = jnp.moveaxis(x0_idx, 0, -2)  # (*batch, T', k_S, T)

    def per_i_core(c_kTD, idx_kT):
        # c_kTD: (k_S, T, D), idx_kT: (k_S, T) -> (n_inputs, D)
        flat_idx = idx_kT.reshape(-1)
        return jax.vmap(
            lambda c_kT: jax.ops.segment_sum(
                c_kT.reshape(-1), flat_idx, num_segments=n_inputs
            ),
            in_axes=-1,
            out_axes=-1,
        )(c_kTD)

    # Wrap with (n_batch + 1) vmaps to consume (*batch, T') as leading axes.
    scatter = per_i_core
    for _ in range(n_batch + 1):
        scatter = jax.vmap(scatter)

    dY_dS = scatter(contrib_r, idx_r)  # (*batch, T', n_inputs, D)
    dY_dS = jnp.moveaxis(dY_dS, -2, 0)  # (n_inputs, *batch, T', D)
    dY = S_dv + dY_dS

    # ---- laplacian ----
    lap_S_v = jnp.matmul(S.laplacian, v.x)
    S_lap_v = jnp.matmul(S.x, v.laplacian)

    # cross[..., i, d] =
    #  = Σ_{k,j} dS.data[k, ..., i, j] · dv.data[x0_idx[k, ..., i, j], ..., j, d]
    def gather_2d(x0_idx_2d, v_jac_2d):
        # x0_idx_2d: (k_S, T', T), v_jac_2d: (n_inputs, T, D) -> (k_S, T', T, D)
        arange_T = jnp.arange(x0_idx_2d.shape[-1])
        return v_jac_2d[x0_idx_2d, arange_T[None, None, :], :]

    gather_fn = gather_2d
    for _ in range(n_batch):
        # batch axis sits at position 1 in both (k_S | n_inputs are at 0).
        gather_fn = jax.vmap(gather_fn, in_axes=(1, 1), out_axes=1)

    dv_gather = gather_fn(x0_idx, v.jacobian.data)  # (k_S, *batch, T', T, D)
    cross = jnp.einsum('k...ij,k...ijd->...id', S.jacobian.data, dv_gather)

    lap_Y = lap_S_v + 2.0 * cross + S_lap_v
    return FwdLaplArray(Y_x, FwdJacobian.from_dense(dY), lap_Y)


def _sparse_matmul_laplacian(args, kwargs, sparsity_threshold):
    """Custom forward-Laplacian rule for `Y = S @ v` with three branches:

    weak S, weak v     -> mul + reduce_sum rewrite (sparsity preserved end-to-end)
    weak S, dense v    -> hand-written rule that exploits dS's sparsity in the
                            dS @ v half of dY and in the dS·dv cross-term of lap Y
    anything else      -> plain matmul (folx's existing dense path is near-optimal)
    """
    S, v = args
    if not (isinstance(S, FwdLaplArray) and isinstance(v, FwdLaplArray)):
        return forward_laplacian(
            jnp.matmul, sparsity_threshold=sparsity_threshold, disable_jit=True
        )(*args)
    if S.jacobian.weak and v.jacobian.weak:
        return forward_laplacian(
            _mulsum, sparsity_threshold=sparsity_threshold, disable_jit=True
        )(*args)
    if S.jacobian.weak and not v.jacobian.weak:
        return _asymmetric_lap_rule(S, v)
    return forward_laplacian(
        jnp.matmul, sparsity_threshold=sparsity_threshold, disable_jit=True
    )(*args)


@jax.jit
def sparse_matmul(S, x):
    return jnp.matmul(S, x)


register_function('sparse_matmul', _sparse_matmul_laplacian)
