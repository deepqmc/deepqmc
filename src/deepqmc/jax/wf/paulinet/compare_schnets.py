from deepqmc.jax.wf.paulinet.schnet import SchNet as SchNet
from deepqmc.wf.paulinet.schnet import ElectronicSchNet
from deepqmc import Molecule
from deepqmc.jax.molecule import Molecule as jMolecule
import torch
import jax.numpy as jnp
from jax import random
from deepqmc.physics import pairwise_self_distance, pairwise_distance

torch.manual_seed(0)
rng = random.PRNGKey(0)

coords = [[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]
charges = [10, 10]
mol = Molecule(torch.tensor(coords), torch.tensor(charges), 0, 0)
jmol = jMolecule(coords, charges, 0, 0)
nelec = int(mol.charges.sum() - mol.charge)
n_up = (nelec + mol.spin) // 2
n_down = (nelec - mol.spin) // 2
n_nuc = len(mol.charges)
rs = torch.rand(1000, nelec, 3)
jrs = jnp.array(rs)

embedding_dim = 65
dist_feat_dim = 16  # dist_feat_dim and kernel_dim has to be close to equal because log scaling layer widths are not yet implemented in jax version
kernel_dim = 17  # dist_feat_dim and kernel_dim has to be close to equal because log scaling layer widths are not yet implemented in jax version
n_interactions = 3
schnet = ElectronicSchNet(
    n_up,
    n_down,
    n_nuc,
    embedding_dim,
    dist_feat_dim=dist_feat_dim,
    n_interactions=n_interactions,
    kernel_dim=kernel_dim,
)
jschnet = SchNet(
    rng, jmol, embedding_dim, kernel_dim, dist_feat_dim, n_interactions=n_interactions
)


def set_mlp_params(n_layers, torch_module, haiku_params):
    for i in range(n_layers):
        torch_name = f'linear{i+1}'
        haiku_params[f'mlp/~/linear_{i}']['w'] = jnp.array(
            getattr(torch_module, torch_name).weight.detach().T
        )
        if i < n_layers - 1:
            haiku_params[f'mlp/~/linear_{i}']['b'] = jnp.array(
                getattr(torch_module, torch_name).bias.detach()
            )
    return haiku_params


# Copy parameters from torch version to JAX version
jschnet.graph_builder.nuc_embeddings = jnp.array(schnet.Y.weight.detach())
jschnet.graph_builder.elec_embeddings = jnp.array(
    schnet.X.weight.detach().repeat(nelec, 1)
)
for l in range(n_interactions):
    jschnet.layers[l].w_nuc_params = set_mlp_params(
        2, schnet.layers[l].w.n, jschnet.layers[l].w_nuc_params
    )
    jschnet.layers[l].w_same_params = set_mlp_params(
        2, schnet.layers[l].w.same, jschnet.layers[l].w_same_params
    )
    jschnet.layers[l].w_anti_params = set_mlp_params(
        2, schnet.layers[l].w.anti, jschnet.layers[l].w_anti_params
    )
    jschnet.layers[l].g_nuc_params = set_mlp_params(
        1, schnet.layers[l].g.n, jschnet.layers[l].g_nuc_params
    )
    jschnet.layers[l].g_same_params = set_mlp_params(
        1, schnet.layers[l].g.same, jschnet.layers[l].g_same_params
    )
    jschnet.layers[l].g_anti_params = set_mlp_params(
        1, schnet.layers[l].g.anti, jschnet.layers[l].g_anti_params
    )
    jschnet.layers[l].h_params = set_mlp_params(
        1, schnet.layers[l].h, jschnet.layers[l].h_params
    )


dists_elec = pairwise_self_distance(rs, full=True)
dists_nuc = pairwise_distance(rs, mol.coords)
torch_result = jnp.array(schnet(dists_elec, dists_nuc)[0].detach())
jax_result = jschnet(jrs)
print(
    'difference:'
    f' {jnp.abs(jax_result - torch_result).sum()/len(jax_result.reshape(-1))}'
)
