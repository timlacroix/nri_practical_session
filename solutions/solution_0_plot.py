from matplotlib import pyplot as plt
import numpy as np

plt.style.use('seaborn-notebook')
plt.figure(dpi=150)


num_atoms = 5
off_diag_idx = np.ravel_multi_index(
    np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
    [num_atoms, num_atoms]
)
for atoms, edges in loaders['train']:
    # edges contains the off-diagonal elements of the interaction matrix
    interactions = np.reshape(np.zeros((num_atoms, num_atoms)), [-1, 25])
    interactions[0][off_diag_idx] = edges
    interactions = np.reshape(interactions, [5,5])
    # now interactions_{i,j} contains whether or not i and j interact

    timesteps = atoms.shape[2]
    atoms_coord = []
    for atom in range(num_atoms):
        this_atom = []
        for t in range(timesteps):
            datum = atoms[0][atom][t]
            this_atom.append(datum.tolist())
        this_atom = np.array(this_atom)
        plt.scatter(this_atom[:, 0], this_atom[:, 1], s=3*np.sqrt(np.array(range(timesteps))), alpha=0.5)
        atoms_coord.append(this_atom)
    
    for atom_a in range(num_atoms):
        for atom_b in range(atom_a + 1, num_atoms):
            if interactions[atom_a, atom_b] == 1:
                for d1, d2 in zip(atoms_coord[atom_a], atoms_coord[atom_b]):
                    plt.plot(
                        [d1[0], d2[0]],
                        [d1[1], d2[1]],
                        'k-',
                        linewidth = 1,
                        alpha=0.2
                    )
                    
    
    break
plt.show()
