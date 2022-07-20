from ..utils import (
    batch_laplacian,
    electronic_potential,
    nuclear_energy,
    nuclear_potential,
)

all__ = ()


class MolecularHamiltonian:
    def __init__(self, mol):
        self.mol = mol

    def local_energy(self, wf, return_grad=False):
        def loc_ene(rs, mol=self.mol):
            log_wf = lambda x: wf(x)[1]
            lap_log_psis, quantum_force = batch_laplacian(log_wf, return_grad=True)(rs)
            Es_kin = -0.5 * (lap_log_psis + (quantum_force**2).sum(axis=(-2, -1)))
            Es_nuc = nuclear_energy(mol)
            Vs_nuc = nuclear_potential(rs, mol)
            Vs_el = electronic_potential(rs)
            Es_loc = Es_kin + Vs_nuc + Vs_el + Es_nuc
            result = (Es_loc, quantum_force) if return_grad else Es_loc
            return result

        return loc_ene
