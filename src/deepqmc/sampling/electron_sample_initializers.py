from typing import Protocol

import jax
import jax.numpy as jnp

from ..physics import pairwise_distance
from ..types import KeyArray
from ..utils import argmax_random_choice


class ElectronSampleInitializer(Protocol):
    r"""Protocol for electron sample initializers.

    These functions should take a nuclear configuration (charges and coordinates),
    along with the desired number of up and down spin electrons, and return an initial
    guess for the positions of these electrons. The returned positions will typically
    be used to initialize the walkers of an MCMC simulation, Consequently they should be
    not be too far from the equilibrium distribution of electrons for the given nuclear
    configuration. In other words, equilibrating the MCMC chains initialized with these
    electron positions should not take too long.

    Args:
        rng (~deepqmc.types.KeyArray): A random number generator seed.
        charges (jax.Array): The atomic number of the nuclei.
        n_valence (jax.Array): The number of valence electrons for each atom. Without
            ECPs this equals the atomic number for each atom.
        nuclear_coordinates (~jax.Array): The nuclear coordinates.
        n_up (int): The number of spin-up electrons.
        n_down (int): The number of spin-down electrons.
    """

    def __call__(
        self,
        rng: KeyArray,
        charges: jax.Array,
        n_valence: jax.Array,
        nuclear_coordinates: jax.Array,
        n_up: int,
        n_down: int,
    ) -> jax.Array: ...


def assign_electrons_to_nuclei(
    rng: KeyArray, n_valence: jax.Array, n_up: int, n_down: int
):
    r"""Assign electrons to one of the nuclei.

    This function determines how many electrons should be distributed around each atom.

    Args:
        rng (~deepqmc.types.KeyArray): Random number generator seed.
        n_valence (jax.Array): The number of valence electrons for each atom. Without
            ECPs this equals the atomic number for each atom.
        n_up (int): The number of spin-up electrons.
        n_down (int): The number of spin-down electrons.

    Returns:
        jax.Array: The number of electrons assigned to each atom, shape:
            ``[len(n_valence)]``.
    """
    charge = n_valence.sum() - n_up - n_down
    valence_electrons = jnp.array(n_valence) - charge / len(n_valence)
    electrons_of_atom = jnp.floor(valence_electrons).astype(int)

    def cond_fn(value):
        _, electrons_of_atom = value
        return (sum(n_valence) - charge - electrons_of_atom.sum()) > 0

    def body_fn(value):
        rng, electrons_of_atom = value
        rng, rng_categorical = jax.random.split(rng)
        atom_idx = jax.random.categorical(
            rng_categorical, valence_electrons - electrons_of_atom, shape=()
        )
        electrons_of_atom = electrons_of_atom.at[atom_idx].add(1)
        return rng, electrons_of_atom

    _, electrons_of_atom = jax.lax.while_loop(
        cond_fn, body_fn, (rng, electrons_of_atom)
    )
    return electrons_of_atom


def assign_spins_to_nuclei(
    rng: KeyArray,
    electrons_of_nuclei: jax.Array,
    nuclear_coordinates: jax.Array,
    n_up: int,
    n_down: int,
) -> tuple[jax.Array, jax.Array]:
    r"""Assign the spins of electrons around each nucleus.

    Given the number of electrons around each nucleus, the nuclear coordinates, and the
    total number of spin-up and spin-down electrons, this function assigns the spins of
    the electrons around each nucleus. The assignment is made via a heuristic that tries
    to
        - minimize the difference between the number of spin-up and spin-down electrons
            around each nucleus
        - mimic covalent bonds by assigning opposite spin unpaired electrons to
            neighboring nuclei

    Args:
        rng (~deepqmc.types.KeyArray): Random number generator seed.
        electrons_of_nuclei (jax.Array): The number of electrons around each nucleus.
        nuclear_coordinates (jax.Array): The nuclear coordinates.
        n_up (int): The number of spin-up electrons.
        n_down (int): The number of spin-down electrons.
    """
    up, down = jnp.zeros_like(electrons_of_nuclei), jnp.zeros_like(electrons_of_nuclei)
    # try to distribute electron pairs evenly across atoms

    def pair_cond_fn(value):
        i, *_ = value
        return i < jnp.max(electrons_of_nuclei)

    def pair_body_fn(value):
        i, up, down = value
        mask = electrons_of_nuclei >= 2 * (i + 1)
        increment = jnp.where(mask & (mask.sum() + down.sum() <= n_down), 1, 0)
        up = up + increment
        down = down + increment
        return i + 1, up, down

    _, up, down = jax.lax.while_loop(pair_cond_fn, pair_body_fn, (0, up, down))

    # distribute remaining electrons such that opposite spin electrons
    # end up close in an attempt to mimic covalent bonds
    dists = (
        pairwise_distance(nuclear_coordinates, nuclear_coordinates)
        .at[jnp.diag_indices(len(nuclear_coordinates))]
        .set(jnp.inf)
    )
    nearest_neighbor_indices = jnp.argsort(dists)

    def spin_cond_fn(value):
        _, _, up, down = value
        return (up + down < electrons_of_nuclei).any()

    def spin_body_fn(value):
        i, center, up, down = value
        is_down = (i % 2) & (down.sum() < n_down)
        up = up.at[center].add(1 - is_down)
        down = down.at[center].add(is_down)
        ordering = nearest_neighbor_indices[center]
        ordered_has_remainder = (electrons_of_nuclei - up - down)[ordering] > 0
        first_ordered_has_remainder = jnp.argmax(ordered_has_remainder)
        center = ordering[first_ordered_has_remainder]
        return i + 1, center, up, down

    center = argmax_random_choice(rng, electrons_of_nuclei - up - down)
    *_, up, down = jax.lax.while_loop(
        spin_cond_fn, spin_body_fn, (jnp.array(0), center, up, down)
    )

    return up, down


def convert_to_nucleus_idx_representation(
    electron_count_per_atom: jax.Array, total_count: int
) -> jax.Array:
    r"""Convert the count of electrons around nuclei to a nucleus index representation.

    Args:
        electron_count_per_atom (jax.Array): The number of electrons around each
            nucleus, shape ``[n_nucleus]``.
        total_count (int): The total number of electrons, should equal to the sum of the
            entries in ``electron_count_per_atom``.

    Returns:
        jax.Array: For each electron, the index of its nucleus, shape ``[total_count]``.
    """
    return (
        jnp.cumsum(electron_count_per_atom)[:, None] <= jnp.arange(total_count)
    ).sum(axis=0)


class AtomCenteredDistribution(Protocol):
    r"""Protocol for electron distributions centered on nuclei."""

    def __call__(
        self, rng: KeyArray, charges: jax.Array, counts_per_nucleus: jax.Array
    ) -> jax.Array: ...


class SingleGaussianDistribution(AtomCenteredDistribution):
    r"""Distribution with a single Gaussian around each nucleus."""

    def __init__(self, scale: float):
        self.scale = scale

    def __call__(
        self, rng: KeyArray, charges: jax.Array, counts_per_nucleus: jax.Array
    ) -> jax.Array:
        std = self.scale * jnp.sqrt(charges)[..., None]
        return std * jax.random.normal(rng, (len(charges), 3))


class ShellBasedDistribution(AtomCenteredDistribution):
    r"""Distribution with an atomic shell structure."""

    def sample_from_shell(
        self, rng: KeyArray, zeta: jax.Array, shape: tuple[int, ...]
    ) -> jax.Array:
        r"""Draw samples from the distribution :math:`r \sim \exp(-2 \zeta ||r||)`."""

        rng_r, rng_direction = jax.random.split(rng)
        distance = jax.random.exponential(rng_r, shape) / (2 * zeta)
        rot_m = jax.random.orthogonal(rng_direction, 3, shape)
        direction = rot_m[..., 0]
        return distance[..., None] * direction

    def spin_idxs_from_spin_counts(
        self, spin_counts: jax.Array, total_spins: int
    ) -> jax.Array:
        return jnp.arange(total_spins) - jnp.cumsum(
            jnp.zeros(total_spins).at[jnp.cumsum(spin_counts)].set(spin_counts)
        )

    def shell_zeta_factor_from_spin_idx(self, spin_idx: jax.Array) -> jax.Array:
        shell_idx_one = jnp.ones_like(spin_idx, dtype=int)
        return jnp.where(
            spin_idx < 1,
            shell_idx_one / 1,
            jnp.where(
                spin_idx < 5,
                shell_idx_one / 2,
                jnp.where(
                    spin_idx < 9,
                    shell_idx_one / 3,
                    shell_idx_one / 4,
                ),
            ),
        )

    def __call__(
        self, rng: KeyArray, charges: jax.Array, counts_per_nucleus: jax.Array
    ) -> jax.Array:
        total_count = len(charges)
        positions = jnp.zeros((total_count, 3))
        zetas = charges * self.shell_zeta_factor_from_spin_idx(
            self.spin_idxs_from_spin_counts(counts_per_nucleus, total_count)
        )

        def scan_body(_carry, x: tuple[KeyArray, jax.Array]):
            rng, zeta = x
            return _carry, self.sample_from_shell(rng, zeta, ())

        _, positions = jax.lax.scan(
            scan_body, None, (jax.random.split(rng, len(zetas)), zetas)
        )
        return positions


class AtomCenteredElectronInitializer(ElectronSampleInitializer):
    r"""Electron initializer that places electrons around nuclei."""

    def __init__(self, atom_centered_distribution: AtomCenteredDistribution):
        self.atom_centered_distribution = atom_centered_distribution

    def __call__(
        self,
        rng: KeyArray,
        charges: jax.Array,
        n_valence: jax.Array,
        nuclear_coordinates: jax.Array,
        n_up: int,
        n_down: int,
    ) -> jax.Array:
        rng_electron_assignment, rng_spin_assignment, rng_up_pos, rng_down_pos = (
            jax.random.split(rng, 4)
        )
        electrons_of_nuclei = assign_electrons_to_nuclei(
            rng_electron_assignment, n_valence, n_up, n_down
        )
        up_of_nuclei, down_of_nuclei = assign_spins_to_nuclei(
            rng_spin_assignment, electrons_of_nuclei, nuclear_coordinates, n_up, n_down
        )
        up_idxs = convert_to_nucleus_idx_representation(up_of_nuclei, n_up)
        down_idxs = convert_to_nucleus_idx_representation(down_of_nuclei, n_down)
        up_positions = nuclear_coordinates[up_idxs] + self.atom_centered_distribution(
            rng_up_pos, charges[up_idxs], up_of_nuclei
        )
        down_positions = nuclear_coordinates[
            down_idxs
        ] + self.atom_centered_distribution(
            rng_down_pos, charges[down_idxs], down_of_nuclei
        )
        return jnp.concatenate([up_positions, down_positions])
