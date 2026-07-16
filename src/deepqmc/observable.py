from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from functools import partial
from typing import Any, Optional, Type
from typing_extensions import Self

import jax
import jax.numpy as jnp

from .force import (
    antithetic_wrapper,
    evaluate_finite_difference_force,
    evaluate_hf_force_ac_zb,
    evaluate_hf_force_ac_zv,
    evaluate_hf_force_ac_zvq,
    evaluate_hf_force_ac_zvqzb,
    evaluate_hf_force_ac_zvzb,
    evaluate_hf_force_ac_zvzbq,
    evaluate_hf_force_bare,
)
from .geom.coordinate_transform import InvertibleCoordinateTransform
from .hamil import MolecularHamiltonian
from .parallel import (
    all_device_max,
    all_device_mean,
    all_device_min,
    all_device_std,
    pmap,
    split_on_devices,
)
from .physics import evaluate_spin
from .types import (
    DataDict,
    Energy,
    KeyArray,
    ParametrizedWaveFunction,
    Params,
    PhysicalConfiguration,
    Psi,
    Stats,
)

__all__ = ['ObservableMonitor', 'EnergyMonitor', 'WaveFunctionMonitor']


def rng_wrapper(observable_fn_factory):
    def wrapped_observable_fn_factory(*args, **kwargs):
        observable_fn = observable_fn_factory(*args, **kwargs)

        def wrapped_observable_fn(rng: KeyArray, *args, **kwargs):
            return observable_fn(*args, **kwargs)

        return wrapped_observable_fn

    return wrapped_observable_fn_factory


def compute_mean_and_std(
    name: str, observable_samples: jax.Array, axis: int = -1
) -> dict[str, jax.Array]:
    return {
        f'{name}/mean': jnp.mean(observable_samples, axis),
        f'{name}/std': jnp.std(observable_samples, axis),
    }


class ObservableMonitor:
    r"""Base class for observable monitors evaluated during training or inference.

    An :class:`ObservableMonitor` encapsulates a physical observable (e.g. forces,
    spin) that is computed periodically from wave function samples.  The lifecycle
    has two stages:

    1. **Construction** — sets the sampling frequency and whether raw samples should
       be stored alongside the statistics.
    2. **Finalization** — :meth:`finalize` is called once the Hamiltonian and wave
       function are known; subclasses override it to build :attr:`observable_fn`.
       After finalization the monitor is ready to be called.

    Subclasses must set the class attribute :attr:`name` and override
    :meth:`finalize` to populate :attr:`observable_fn`.  Set
    :attr:`requires_energy` to ``True`` when the observable function needs the
    local energies as an additional input.

    Args:
        save_samples (bool): if ``True``, the raw per-sample observable values
            are included in the returned statistics dictionary under the key
            ``'<name>/samples'``.
        period (int): number of training steps between consecutive evaluations;
            must be at least 1.
    """

    name: str
    save_samples: bool
    period: int
    observable_fn: Optional[Callable] = None
    requires_energy: bool = False

    def __init__(self, save_samples: bool, period: int):
        assert period > 0
        self.save_samples = save_samples
        self.period = period

    def finalize(
        self, hamil: MolecularHamiltonian, wf: ParametrizedWaveFunction
    ) -> Self:
        r"""Bind the monitor to a specific Hamiltonian and wave function.

        Called once before training begins.  The default implementation returns
        ``self`` unchanged; subclasses override this to construct
        :attr:`observable_fn` from ``hamil`` and ``wf``.

        Args:
            hamil (~deepqmc.hamil.MolecularHamiltonian): the Hamiltonian of the
                physical system.
            wf (~deepqmc.types.ParametrizedWaveFunction): the parametrized wave
                function used during training.

        Returns:
            ~deepqmc.observable.ObservableMonitor: the finalized monitor (``self``).
        """
        return self

    @partial(pmap, static_broadcasted_argnums=(0,))
    def compute_observable(
        self,
        rng: KeyArray,
        params: Params,
        phys_conf: PhysicalConfiguration,
        psi: Psi,
        local_energy: Energy,
        psi_ratios: Optional[jax.Array],
    ) -> tuple[Any, Stats]:
        assert self.observable_fn is not None, 'call ObservableMonitor.finalize first'
        if not self.requires_energy:
            observable_samples = jax.vmap(
                jax.vmap(jax.vmap(self.observable_fn, (0, None, 0))), (0, None, 0)
            )(rng, params, phys_conf)
        else:
            observable_samples = jax.vmap(
                jax.vmap(jax.vmap(self.observable_fn, (0, None, 0, 0, None))),
                (0, None, 0, 0, 0),
            )(rng, params, phys_conf, local_energy, local_energy.mean(-1))
        stats = compute_mean_and_std(self.name, observable_samples, axis=2)
        return observable_samples, stats

    def __call__(
        self,
        step: int,
        rng: KeyArray,
        params: Params,
        phys_conf: PhysicalConfiguration,
        psi: Psi,
        local_energy: jax.Array | None,
        psi_ratios: Optional[jax.Array],
    ) -> Stats:
        r"""Evaluate the observable at the current training step.

        Returns an empty dictionary on steps that are not multiples of
        :attr:`period`.  Otherwise distributes the RNG across devices,
        calls :meth:`compute_observable`, and optionally attaches the raw
        samples to the statistics dictionary.

        Args:
            step (int): current training step index.
            rng (~deepqmc.types.KeyArray): RNG key for stochastic observables.
            params (~deepqmc.types.Params): current wave function parameters.
            phys_conf (~deepqmc.types.PhysicalConfiguration): electron and
                nuclear configurations.
            psi (~deepqmc.types.Psi): current wave function values.
            local_energy (~jax.Array | None): per-sample local energies; may
                be ``None`` if not yet computed.
            psi_ratios (Optional[~jax.Array]): wave function ratios for
                multi-state calculations, or ``None``.

        Returns:
            ~deepqmc.types.Stats: a statistics dictionary, or ``{}`` if this
            step is skipped.
        """
        if step % self.period:
            return {}
        rng = jnp.array(split_on_devices(rng, phys_conf.batch_shape))[0]
        observable_samples, stats = self.compute_observable(
            rng, params, phys_conf, psi, local_energy, psi_ratios
        )
        if self.save_samples and observable_samples is not None:
            stats |= {f'{self.name}/samples': observable_samples}
        return stats


class SpinMonitor(ObservableMonitor):
    r"""Monitor the total spin expectation value :math:`\langle S^2 \rangle`."""

    name: str = 'spin'

    def finalize(self, hamil: MolecularHamiltonian, wf) -> Self:
        self.observable_fn = rng_wrapper(evaluate_spin)(hamil, wf)
        return self


class BaseForceMonitor(ObservableMonitor, ABC):
    r"""Abstract base class for Hellmann-Feynman force monitors with optional coordinate transform."""

    def __init__(
        self,
        save_samples: bool,
        period: int,
        coordinate_transform: Optional[InvertibleCoordinateTransform] = None,
    ):
        super().__init__(save_samples, period)
        self.coordinate_transform = coordinate_transform

    def finalize(
        self, hamil: MolecularHamiltonian, wf: ParametrizedWaveFunction
    ) -> Self:
        self.observable_fn = self.evaluate_hf_force(
            hamil, wf, self.coordinate_transform
        )
        return self

    @staticmethod
    @abstractmethod
    def evaluate_hf_force(
        hamil: MolecularHamiltonian,
        wf: ParametrizedWaveFunction,
        coordinate_transform: Optional[InvertibleCoordinateTransform] = None,
    ) -> Callable:
        pass


class BaseForceMonitorNotRequiringEnergy(BaseForceMonitor, ABC):
    r"""Abstract base for force monitors that do not require local energies as input."""

    requires_energy = False

    @staticmethod
    @abstractmethod
    def evaluate_hf_force(
        hamil: MolecularHamiltonian,
        wf: ParametrizedWaveFunction,
        coordinate_transform: Optional[InvertibleCoordinateTransform] = None,
    ) -> Callable[[KeyArray, Params, PhysicalConfiguration], jax.Array]:
        pass


class BaseForceMonitorRequiringEnergy(BaseForceMonitor, ABC):
    r"""Abstract base for force monitors that require local energies as input."""

    requires_energy = True

    @staticmethod
    @abstractmethod
    def evaluate_hf_force(
        hamil: MolecularHamiltonian,
        wf: ParametrizedWaveFunction,
        coordinate_transform: Optional[InvertibleCoordinateTransform] = None,
    ) -> Callable[[KeyArray, Params, PhysicalConfiguration, Energy, Energy], jax.Array]:
        pass


class BareForceMonitor(BaseForceMonitorNotRequiringEnergy):
    r"""Monitor bare Hellmann-Feynman forces without variance reduction."""

    name: str = 'hf_force_bare'

    @staticmethod
    def evaluate_hf_force(
        hamil: MolecularHamiltonian,
        wf: ParametrizedWaveFunction,
        coordinate_transform: Optional[InvertibleCoordinateTransform] = None,
    ) -> Callable[[KeyArray, Params, PhysicalConfiguration], jax.Array]:
        return evaluate_hf_force_bare(hamil, wf, coordinate_transform)


class BareForceAntiMonitor(BareForceMonitor):
    r"""Monitor bare Hellmann-Feynman forces with antithetic-sampling variance reduction."""

    name: str = 'hf_force_bare_anti'

    def __init__(
        self,
        save_samples: bool,
        period: int,
        coordinate_transform: Optional[InvertibleCoordinateTransform] = None,
        cutoff: float = 0.3,
    ):
        super().__init__(save_samples, period, coordinate_transform)
        self.cutoff = cutoff

    def finalize(
        self, hamil: MolecularHamiltonian, wf: ParametrizedWaveFunction
    ) -> Self:
        self.observable_fn = antithetic_wrapper(
            self.evaluate_hf_force(hamil, wf, self.coordinate_transform),  # type: ignore
            wf,
            self.cutoff,
        )
        return self

    @staticmethod
    def evaluate_hf_force(
        hamil: MolecularHamiltonian,
        wf: ParametrizedWaveFunction,
        coordinate_transform: Optional[InvertibleCoordinateTransform] = None,
    ) -> Callable[[KeyArray, Params, PhysicalConfiguration], jax.Array]:
        return evaluate_hf_force_bare(hamil, wf, coordinate_transform)


class ACZVForceMonitor(BaseForceMonitorRequiringEnergy):
    r"""Monitor HF forces using the AC-ZV estimator [Assaraf03]_."""

    name: str = 'hf_force_ac_zv'

    @staticmethod
    def evaluate_hf_force(
        hamil: MolecularHamiltonian,
        wf: ParametrizedWaveFunction,
        coordinate_transform: Optional[InvertibleCoordinateTransform] = None,
    ) -> Callable[[KeyArray, Params, PhysicalConfiguration, Energy, Energy], jax.Array]:
        return evaluate_hf_force_ac_zv(hamil, wf, coordinate_transform)


class ACZVZBForceMonitor(BaseForceMonitorRequiringEnergy):
    r"""Monitor HF forces using the AC-ZV-ZB estimator [Assaraf03]_."""

    name: str = 'hf_force_ac_zvzb'

    @staticmethod
    def evaluate_hf_force(
        hamil: MolecularHamiltonian,
        wf: ParametrizedWaveFunction,
        coordinate_transform: Optional[InvertibleCoordinateTransform] = None,
    ) -> Callable[[KeyArray, Params, PhysicalConfiguration, Energy, Energy], jax.Array]:
        return evaluate_hf_force_ac_zvzb(hamil, wf, coordinate_transform)


class ACZBForceMonitor(BaseForceMonitorRequiringEnergy):
    r"""Monitor HF forces using the AC-ZB estimator [Assaraf03]_."""

    name: str = 'hf_force_ac_zb'

    @staticmethod
    def evaluate_hf_force(
        hamil: MolecularHamiltonian,
        wf: ParametrizedWaveFunction,
        coordinate_transform: Optional[InvertibleCoordinateTransform] = None,
    ) -> Callable[[KeyArray, Params, PhysicalConfiguration, Energy, Energy], jax.Array]:
        return evaluate_hf_force_ac_zb(hamil, wf, coordinate_transform)


class ACZVQForceMonitor(BaseForceMonitorNotRequiringEnergy):
    r"""Monitor HF forces using the AC-ZVQ estimator [Assaraf03]_; incompatible with ECPs."""

    name: str = 'hf_force_ac_zvq'

    def finalize(
        self, hamil: MolecularHamiltonian, wf: ParametrizedWaveFunction
    ) -> Self:
        assert not jnp.any(hamil.ecp_mask), 'Use ACZV for forces with pseudo-potentials'
        return super().finalize(hamil, wf)

    @staticmethod
    def evaluate_hf_force(
        hamil: MolecularHamiltonian,
        wf: ParametrizedWaveFunction,
        coordinate_transform: Optional[InvertibleCoordinateTransform] = None,
    ):
        return rng_wrapper(evaluate_hf_force_ac_zvq)(hamil, wf, coordinate_transform)


class ACZVQForceAntiMonitor(ACZVQForceMonitor):
    r"""Monitor HF forces using AC-ZVQ with antithetic sampling; incompatible with ECPs."""

    name: str = 'hf_force_ac_zvq_anti'

    def __init__(
        self,
        save_samples: bool,
        period: int,
        coordinate_transform: Optional[InvertibleCoordinateTransform] = None,
        cutoff: float = 0.3,
    ):
        super().__init__(save_samples, period, coordinate_transform)
        self.cutoff = cutoff

    def finalize(
        self, hamil: MolecularHamiltonian, wf: ParametrizedWaveFunction
    ) -> Self:
        assert not jnp.any(
            hamil.ecp_mask
        ), 'antithetic ACZVQ forces are not implemented with ECPs'
        self.observable_fn = antithetic_wrapper(
            self.evaluate_hf_force(hamil, wf, self.coordinate_transform),
            wf,
            self.cutoff,
        )
        return self


class ACZVZBQForceMonitor(BaseForceMonitorRequiringEnergy):
    r"""Monitor HF forces using the AC-ZV-ZB-Q estimator [Assaraf03]_; incompatible with ECPs."""

    name: str = 'hf_force_ac_zvzbq'

    def finalize(
        self, hamil: MolecularHamiltonian, wf: ParametrizedWaveFunction
    ) -> Self:
        assert not jnp.any(
            hamil.ecp_mask
        ), 'Use ACZVZB for forces with pseudo-potentials'
        return super().finalize(hamil, wf)

    @staticmethod
    def evaluate_hf_force(
        hamil: MolecularHamiltonian,
        wf: ParametrizedWaveFunction,
        coordinate_transform: Optional[InvertibleCoordinateTransform] = None,
    ):
        return rng_wrapper(evaluate_hf_force_ac_zvzbq)(hamil, wf, coordinate_transform)


class ACZVQZBForceMonitor(BaseForceMonitorRequiringEnergy):
    r"""Monitor HF forces using the AC-ZVQ-ZB hybrid estimator; incompatible with ECPs."""

    name: str = 'hf_force_ac_zvqzb'

    def finalize(self, hamil: MolecularHamiltonian, wf) -> Self:
        assert not jnp.any(
            hamil.ecp_mask
        ), 'Use ACZVZB for forces with pseudo-potentials'
        return super().finalize(hamil, wf)

    @staticmethod
    def evaluate_hf_force(
        hamil: MolecularHamiltonian,
        wf: ParametrizedWaveFunction,
        coordinate_transform: Optional[InvertibleCoordinateTransform] = None,
    ):
        return rng_wrapper(evaluate_hf_force_ac_zvqzb)(hamil, wf, coordinate_transform)


class FiniteDifferenceForceMonitor(ObservableMonitor):
    r"""Monitor interatomic forces via a finite-difference scheme."""

    name: str = 'finite_difference_force'

    def __init__(self, save_samples: bool, period: int, h: float = 1e-3):
        super().__init__(save_samples, period)
        self.h = h

    def finalize(self, hamil: MolecularHamiltonian, wf) -> Self:
        self.observable_fn = evaluate_finite_difference_force(hamil, wf, self.h)
        self.requires_energy = True
        return self


class EnergyMonitor(ObservableMonitor):
    r"""Monitor the local energies during the calculation."""

    name: str = 'local_energy'

    @partial(pmap, static_broadcasted_argnums=(0,))
    def compute_observable(
        self,
        rng: KeyArray,
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
    r"""Monitor wave function ratios between electronic states."""

    name: str = 'psi_ratio'

    @partial(pmap, static_broadcasted_argnums=(0,))
    def compute_observable(
        self,
        rng: KeyArray,
        params: Params,
        phys_conf: PhysicalConfiguration,
        psi: Psi,
        local_energy: Energy,
        psi_ratios: Optional[jax.Array],
    ) -> tuple[jax.Array, DataDict]:
        assert psi_ratios is not None
        return psi_ratios, {}


class ElectronPositionMonitor(ObservableMonitor):
    r"""Monitor the electron positions during training."""

    name: str = 'r'

    @partial(pmap, static_broadcasted_argnums=(0,))
    def compute_observable(
        self,
        rng: KeyArray,
        params: Params,
        phys_conf: PhysicalConfiguration,
        psi: Psi,
        local_energy: Energy,
        psi_ratios: Optional[jax.Array],
    ) -> tuple[jax.Array, DataDict]:
        return phys_conf.r, {}


class NuclearPositionMonitor(ObservableMonitor):
    r"""Monitor the nuclear positions during training."""

    name: str = 'R'

    @partial(pmap, static_broadcasted_argnums=(0,))
    def compute_observable(
        self,
        rng: KeyArray,
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
        rng: KeyArray,
        params: Params,
        phys_conf: PhysicalConfiguration,
        psi: Psi,
        local_energy: Energy,
        psi_ratios: Optional[jax.Array],
    ) -> tuple[Mapping[str, jax.Array], DataDict]:
        return {'sign': psi.sign, 'log': psi.log}, {}


class OscillatorStrengthMonitor(ObservableMonitor):
    r"""Monitor oscillator strengths between electronic states."""

    name: str = 'oscillator_strength'

    @partial(pmap, static_broadcasted_argnums=(0,))
    def compute_observable(
        self,
        rng: KeyArray,
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
        WaveFunctionMonitor(save_samples=True, period=1),
    ]


def observable_monitor_from_name(name: str) -> ObservableMonitor:
    all_obseravble_monitors: set[Type[ObservableMonitor]] = {
        ElectronPositionMonitor,
        NuclearPositionMonitor,
        WaveFunctionMonitor,
        EnergyMonitor,
        SpinMonitor,
        PsiRatioMonitor,
        OscillatorStrengthMonitor,
        BareForceMonitor,
        BareForceAntiMonitor,
        ACZVForceMonitor,
        ACZVQForceMonitor,
        ACZVQForceAntiMonitor,
        ACZVZBForceMonitor,
        ACZBForceMonitor,
        ACZVZBQForceMonitor,
        FiniteDifferenceForceMonitor,
        ACZVQZBForceMonitor,
    }
    for monitor in all_obseravble_monitors:
        if monitor.name == name:
            return monitor(save_samples=True, period=1)
    raise ValueError(f'Unknown observable monitor: {name}')
