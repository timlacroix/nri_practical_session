n_atoms = 5
d = 2
features = torch.FloatTensor([[i] * d for i in range(n_atoms)])

id1 = torch.LongTensor(sum([[i] * n_atoms for i in range(n_atoms)], []))
id2 = torch.LongTensor(sum([list(range(n_atoms)) for i in range(n_atoms)], []))

def v_to_e(x, id1, id2):
    return torch.cat([
        torch.index_select(x, 0, id1),
        torch.index_select(x, 0, id2),
    ], 1)

aggregator = torch.FloatTensor([
    [1./n_atoms if row * n_atoms <= col < (row + 1) * n_atoms else 0 for col in range(n_atoms * n_atoms)]
    for row in range(n_atoms)
])

def e_to_v(x, matrix):
    return matrix @ x

id3 = []
for i in range(n_atoms):
    for j in range(n_atoms):
        if i != j :
            id3.append(i * n_atoms + j)          

def remove_self_edges(features):
    return torch.index_select(features, 0, id3)

print(remove_self_edges(v_to_e(features, id1, id2)))
