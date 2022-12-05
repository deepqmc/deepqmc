import haiku as hk
import jax.numpy as jnp

from ...utils import factorial2

__all__ = ['GTOBasis']


def get_cartesian_angulars(l):
    return [
        (lx, ly, l - lx - ly) for lx in range(l, -1, -1) for ly in range(l - lx, -1, -1)
    ]


class GTOShell(hk.Module):
    def __init__(self, l, coeffs, zetas, name=None):
        super().__init__(name)
        zetas = jnp.asarray(zetas)
        self.ls = jnp.asarray(get_cartesian_angulars(l))
        self.anorms = 1.0 / jnp.sqrt(factorial2(2 * self.ls - 1).prod(axis=-1))
        self.rnorms = (2 * zetas / jnp.pi) ** (3 / 4) * (4 * zetas) ** (l / 2)
        self.coeffs = hk.Linear(
            1,
            with_bias=False,
            w_init=lambda s, d: jnp.asarray(coeffs)[:, None],
            name='coeffs',
        )
        self.zetas = hk.get_parameter('zetas', [len(zetas)], init=lambda s, d: zetas)

    def __call__(self, diffs):
        rs, rs_2 = diffs[..., :3], diffs[..., 3]
        angulars = jnp.power(rs[..., None, :], self.ls).prod(axis=-1)
        exps = self.rnorms * jnp.exp(-jnp.abs(self.zetas * rs_2[..., None]))
        radials = self.coeffs(exps).squeeze(axis=-1)
        phis = self.anorms * angulars * radials[..., None]
        return phis


class GTOBasis(hk.Module):
    r"""Represent a GTO basis of a molecule."""

    def __init__(self, centers, shells):
        super().__init__()
        self.centers = jnp.asarray(centers)
        self.shells = [
            (atom, GTOShell(l, coeff, zeta, name=f'gto_shell_atom{atom}_l{l}'))
            for atom, (l, coeff, zeta) in shells
        ]

    def __call__(self, diffs):
        return jnp.concatenate(
            [shell(diffs[..., idx, :]) for idx, shell in self.shells], axis=-1
        )

    @classmethod
    def from_pyscf(cls, mol):
        r"""Create the input of the constructor from a :class:`~deepqmc.Molecule`.

        Args:
            mol (~deepqmc.Molecule): the molecule to consider.
        """
        assert mol.cart
        centers = mol.atom_coords()
        shells = []
        for i in range(mol.nbas):
            l = mol.bas_angular(i)
            i_atom = mol.bas_atom(i)
            zetas = mol.bas_exp(i)
            coeff_sets = mol.bas_ctr_coeff(i).T
            for coeffs in coeff_sets:
                shells.append((i_atom, (l, coeffs, zetas)))
        return centers, shells
