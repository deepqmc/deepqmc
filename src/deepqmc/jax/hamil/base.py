class Hamiltonian:
    r"""Base class for all Hamiltonian operators."""

    def local_energy(self, wf):
        r"""
        Return a function that calculates the local energy of the wave function.

        Args:
            wf (~jax.wf.WaveFunction): the wave function ansatz.

        Returns:
            :class:`Callable[r, ...]`: a function that evaluates
            the local energy of :data:`wf` at :data:`r`.
        """
        return NotImplementedError
