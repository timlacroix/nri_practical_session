import torch.nn.functional as F

class MLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]
        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        return self.batch_norm(x)



def ids_and_agg(n_atoms, no_self_edges=False):
    n_for_agg = (n_atoms - 1) if no_self_edges else n_atoms
    return (
        torch.cuda.LongTensor(sum([[i] * n_atoms for i in range(n_atoms)], [])),
        torch.cuda.LongTensor(sum([list(range(n_atoms)) for i in range(n_atoms)], [])),
        torch.cuda.FloatTensor([
            [1. / n_for_agg if row * n_for_agg <= col < (row + 1) * n_for_agg else 0
             for col in range(n_for_agg * n_atoms)]
            for row in range(n_atoms)
        ]),
        torch.cuda.LongTensor([
            i for i in range(n_atoms * n_atoms)
            if i not in set([j*n_atoms + j for j in range(n_atoms)])
        ])
    )

class MLPEncoder(nn.Module):
    def __init__(self, n_atoms, n_in, n_hid, n_out, do_prob=0.):
        """
        Given an input of shape [batch_size, num_atoms, num_timesteps, num_dims],
        output a tensor of shape [num_atoms * (num_atoms - 1), n_out] with class logits for each atom-atom interaction edge.
        
        :param n_atoms number of atoms in the simulation
        :param n_in total number of features for one atom, ie, n_dim * n_timesteps
        :param n_hid size of the hidden layer
        :param n_out number of classes to output for the encoder (ie, edge types)
        """
        super(MLPEncoder, self).__init__()
        # mlp1 is f_emb
        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
        # mlp2 is f_e^1 
        self.mlp2 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
        # mlp3 is f_v^1
        self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)
        # mlp4 is f_e^2
        self.mlp4 = MLP(n_hid * 3, n_hid, n_hid, do_prob)
        # fc_out to output the logits of each interaction class
        self.fc_out = nn.Linear(n_hid, n_out)
        
        self.id1, self.id2, self.aggregator, self.id3 = ids_and_agg(n_atoms)
        
    def tile(self, x):
        # v -> e
        return torch.cat([
            torch.index_select(x, 1, self.id1),
            torch.index_select(x, 1, self.id2),
        ], 2)

    def aggregate(self, x):
        # e -> v
        return self.aggregator @ x

    def forward(self, inputs):
        # Input shape: [num_sims, num_atoms, num_timesteps, num_dims]
        x = inputs.view(inputs.size(0), inputs.size(1), -1)
        # New shape: [num_sims, num_atoms, num_timesteps*num_dims]
        
        x = self.mlp1(x)                  # eq 1
        x_skip = self.mlp2(self.tile(x))       # eq 2
        x = self.mlp3(self.aggregate(x_skip))  # eq 3
        x = self.mlp4(torch.cat((self.tile(x), x_skip), dim=2))       # eq 4
        
        logits = self.fc_out(x)
        return torch.index_select(logits, 1, self.id3)       # remove self-edges

