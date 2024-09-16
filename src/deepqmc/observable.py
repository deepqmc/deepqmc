from collections.abc import Callable, Mapping
from functools import partial
from typing import Any, Optional
from typing_extensions import Self

import jax
import jax.numpy as jnp

from .force import (
    evaluate_hf_force_ac_zv,
    evaluate_hf_force_ac_zvq,
    evaluate_hf_force_ac_zvzb,
    evaluate_hf_force_ac_zvzbq,
    evaluate_hf_force_bare,
)
from .hamil import MolecularHamiltonian
from .parallel import (
    all_device_max,
    all_device_mean,
    all_device_min,
    all_device_std,
    pmap,
)
from .physics import evaluate_spin
from .types import (
    DataDict,
    Energy,
    ParametrizedWaveFunction,
    Params,
    PhysicalConfiguration,
    Psi,
    Stats,
)

__all__ = ['default_observable_monitors', 'EnergyMonitor', 'WaveFunctionMonitor']


def compute_mean_and_std(
    name: str, observable_samples: jax.Array, axis: int = -1
) -> dict[str, jax.Array]:
    return {
        f'{name}/mean': jnp.mean(observable_samples, axis),
        f'{name}/std': jnp.std(observable_samples, axis),
    }


class ObservableMonitor:
    name: str
    save_samples: bool
    period: int
    observable_fn: Optional[Callable]

    def __init__(self, save_samples: bool, period: int):
        assert period > 0
        self.save_samples = save_samples
        self.period = period
        self.observable_fn = None
        self.requires_energy = False

    def finalize(
        self, hamil: MolecularHamiltonian, wf: ParametrizedWaveFunction
    ) -> Self:
        return self

    @partial(pmap, static_broadcasted_argnums=(0,))
    def compute_observable(
        self,
        params: Params,
        phys_conf: PhysicalConfiguration,
        psi: Psi,
        local_energy: Energy,
        psi_ratios: Optional[jax.Array],
    ) -> tuple[Any, Stats]:
        assert self.observable_fn is not None, 'call ObservableMonitor.finalize first'
        if not self.requires_energy:
            observable_samples = jax.vmap(
                jax.vmap(jax.vmap(self.observable_fn, (None, 0))), (None, 0)
            )(params, phys_conf)
        else:
            observable_samples = jax.vmap(
                jax.vmap(jax.vmap(self.observable_fn, (None, 0, 0, None))),
                (None, 0, 0, 0),
            )(params, phys_conf, local_energy, local_energy.mean(-1))
        stats = compute_mean_and_std(self.name, observable_samples, axis=2)
        return observable_samples, stats

    def __call__(
        self,
        step: int,
        params: Params,
        phys_conf: PhysicalConfiguration,
        psi: Psi,
        local_energy: jax.Array,
        psi_ratios: Optional[jax.Array],
    ) -> Stats:
        if step % self.period:
            return {}
        observable_samples, stats = self.compute_observable(
            params, phys_conf, psi, local_energy, psi_ratios
        )
        if self.save_samples and observable_samples is not None:
            stats |= {f'{self.name}/samples': observable_samples}
        return stats


class SpinMonitor(ObservableMonitor):
    name: str = 'spin'

    def finalize(self, hamil: MolecularHamiltonian, wf) -> Self:
        self.observable_fn = evaluate_spin(hamil, wf)
        return self


class BareForceMonitor(ObservableMonitor):
    name: str = 'hf_force_bare'

    def finalize(self, hamil: MolecularHamiltonian, wf) -> Self:
        self.observable_fn = evaluate_hf_force_bare(hamil)
        return self


class ACZVForceMonitor(ObservableMonitor):
    name: str = 'hf_force_ac_zv'

    def finalize(self, hamil: MolecularHamiltonian, wf) -> Self:
        self.observable_fn = evaluate_hf_force_ac_zv(hamil, wf)
        return self


class ACZVZBForceMonitor(ObservableMonitor):
    name: str = 'hf_force_ac_zvzb'

    def finalize(self, hamil: MolecularHamiltonian, wf) -> Self:
        self.observable_fn = evaluate_hf_force_ac_zvzb(hamil, wf)
        self.requires_energy = True
        return self


class ACZVQForceMonitor(ObservableMonitor):
    name: str = 'hf_force_ac_zvq'

    def finalize(self, hamil: MolecularHamiltonian, wf) -> Self:
        self.observable_fn = evaluate_hf_force_ac_zvq(hamil, wf)
        return self


class ACZVZBQForceMonitor(ObservableMonitor):
    name: str = 'hf_force_ac_zvzbq'

    def finalize(self, hamil: MolecularHamiltonian, wf) -> Self:
        self.observable_fn = evaluate_hf_force_ac_zvzbq(hamil, wf)
        self.requires_energy = True
        return self


class EnergyMonitor(ObservableMonitor):
    r"""Monitor the local energies during the calculation."""

    name: str = 'local_energy'

    @partial(pmap, static_broadcasted_argnums=(0,))
    def compute_observable(
        self,
        params: Params,
        phys_conf: PhysicalConfiguration,
        psi: Psi,
        local_energy: Energy,
        psi_ratios: Optional[jax.Array],
    ) -> tuple[Energy, Stats]:
        stats = {
            f'{self.name}/mean': all_device_mean(local_energy, axis=-1),
            f'{self.name}/std': all_device_std(local_energy, axis=-1),
            f'{self.name}/min': all_device_min(local_energy, axis=-1),
            f'{self.name}/max': all_device_max(local_energy, axis=-1),
        }
        return local_energy, stats


class PsiRatioMonitor(ObservableMonitor):
    name: str = 'psi_ratio'

    @partial(pmap, static_broadcasted_argnums=(0,))
    def compute_observable(
        self,
        params: Params,
        phys_conf: PhysicalConfiguration,
        psi: Psi,
        local_energy: Energy,
        psi_ratios: Optional[jax.Array],
    ) -> tuple[jax.Array, DataDict]:
        assert psi_ratios is not None
        return psi_ratios, {}


class ElectronPositionMonitor(ObservableMonitor):
    name: str = 'r'

    @partial(pmap, static_broadcasted_argnums=(0,))
    def compute_observable(
        self,
        params: Params,
        phys_conf: PhysicalConfiguration,
        psi: Psi,
        local_energy: Energy,
        psi_ratios: Optional[jax.Array],
    ) -> tuple[jax.Array, DataDict]:
        return phys_conf.r, {}


class NuclearPositionMonitor(ObservableMonitor):
    name: str = 'R'

    @partial(pmap, static_broadcasted_argnums=(0,))
    def compute_observable(
        self,
        params: Params,
        phys_conf: PhysicalConfiguration,
        psi: Psi,
        local_energy: Energy,
        psi_ratios: Optional[jax.Array],
    ) -> tuple[jax.Array, DataDict]:
        return phys_conf.R[..., 0, :, :], {}


class WaveFunctionMonitor(ObservableMonitor):
    r"""Monitor the wave function during the calculation."""

    name: str = 'psi'

    @partial(pmap, static_broadcasted_argnums=(0,))
    def compute_observable(
        self,
        params: Params,
        phys_conf: PhysicalConfiguration,
        psi: Psi,
        local_energy: Energy,
        psi_ratios: Optional[jax.Array],
    ) -> tuple[Mapping[str, jax.Array], DataDict]:
        return {'sign': psi.sign, 'log': psi.log}, {}


class OscillatorStrengthMonitor(ObservableMonitor):
    name: str = 'oscillator_strength'

    @partial(pmap, static_broadcasted_argnums=(0,))
    def compute_observable(
        self,
        params: Params,
        phys_conf: PhysicalConfiguration,
        psi: Psi,
        local_energy: Energy,
        psi_ratios: Optional[jax.Array],
    ) -> tuple[None, DataDict]:
        assert psi_ratios is not None
        sample_size = local_energy.shape[-1] * jax.device_count()

        # excitation energy
        energy_mean = all_device_mean(local_energy, axis=-1)
        energy_err = all_device_std(local_energy, axis=-1) / sample_size**0.5
        ex_energy_mean = energy_mean[None, :] - energy_mean[:, None]
        ex_energy_err = (energy_err**2 + energy_err[:, None] ** 2) ** 0.5

        # dipole strength [molecule_batch_size, electronic_state, electronic_state]
        cd = jnp.sum(-phys_conf.r, axis=-2)[:, None] * psi_ratios[..., None]
        cd_mean = all_device_mean(cd, axis=-2)
        cd_err = all_device_std(cd, axis=-2) / sample_size**0.5
        cd_rel_err = cd_err / cd_mean

        ds_vec = cd_mean * cd_mean.swapaxes(1, 2)
        ds_err_vec = (
            jnp.abs(ds_vec) * (cd_rel_err**2 + cd_rel_err.swapaxes(1, 2) ** 2) ** 0.5
        )

        ds_mean = jnp.sum(ds_vec, axis=-1)
        ds_err = jnp.sum(ds_err_vec**2, axis=-1) ** 0.5

        # oscillator strength
        os_mean = (2 / 3) * ex_energy_mean * ds_mean
        os_err = (
            (2 / 3)
            * jnp.abs(os_mean)
            * ((ex_energy_err / ex_energy_mean) ** 2 + (ds_err / ds_mean) ** 2) ** 0.5
        )

        return None, {
            f'{self.name}/mean': os_mean,
            f'{self.name}/err': os_err,
        }


def default_observable_monitors() -> list[ObservableMonitor]:
    r"""Return a list of default observable monitors."""
    return [
        EnergyMonitor(save_samples=True, period=1),
        WaveFunctionMonitor(save_samples=True, period=1),
    ]
