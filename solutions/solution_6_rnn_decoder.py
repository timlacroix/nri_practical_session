class RNNDecoder(nn.Module):
    """Recurrent decoder module."""

    def __init__(self, n_dims, n_hid, do_prob=0.):
        super(RNNDecoder, self).__init__()
        
        # Linear, Tanh, Dropout, Linear, Tanh
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * n_hid, n_hid), nn.Tanh(), nn.Dropout(do_prob),
            nn.Linear(n_hid, n_hid), nn.Tanh()
        )
        
        # GruCell
        self.gru = nn.GRUCell(input_size=n_dims, hidden_size=n_hid)
        
        # Linear, ReLU, Linear, ReLU, Linear
        self.decoder = nn.Sequential(
            nn.Linear(n_hid, n_hid), nn.ReLU(),
            nn.Linear(n_hid, n_hid), nn.ReLU(),
            nn.Linear(n_hid, n_dims)
        )
        self.n_dims = n_dims
        self.n_hid = n_hid
        
        self.id1, self.id2, self.aggregator, self.id3 = ids_and_agg(n_atoms, True)
        
    def step(self, inputs, edges, hidden):
        """
        Compute one step of the decoder
        :param inputs: a tensor of shape [batch_size x n_atoms x dims]
        :param edges: a tensor of shape  [batch_size x n_edges x edge_type]
        :param hidden: a tensor of shape [batch_size x n_atoms x hidden_size]
        """
        # concatenate the features (equation 1)
        hidden_state = torch.cat([
            torch.index_select(hidden, 1, self.id1),
            torch.index_select(hidden, 1, self.id2)
        ], 2)
        # remove the self edges
        hidden_without_self = torch.index_select(hidden_state, 1, self.id3)
        # multiply by the z, the probability that the edge is active
        transformed = self.edge_mlp(hidden_without_self) * edges[:,:, 1].unsqueeze(2)
        # aggregate
        hidden_state = self.aggregator @ transformed
        
        # compute the next_hidden state with the gru
        next_hidden = self.gru(
            inputs.contiguous().view(-1, self.n_dims),
            hidden_state.view(-1, self.n_hid)
        ).reshape(hidden.shape)
        
        # compute the output
        output = self.decoder(next_hidden) + inputs

        return output, next_hidden

    def forward(self, data, edges, burn_in_steps=1):

        inputs = data.transpose(1, 2).contiguous()

        time_steps = inputs.size(1)

        hidden = torch.zeros(inputs.size(0), inputs.size(2), self.n_hid).cuda()
        pred_all = []

        for step in range(inputs.size(1) - 1):
            # similar as for the LSTM baseline
            if step <= burn_in_steps:
                ins = inputs[:, step, :, :]
            else:
                ins = pred_all[step - 1]

            pred, hidden = self.step(ins, edges, hidden)
            pred_all.append(pred)

        preds = torch.stack(pred_all, dim=1)

        return preds.transpose(1, 2)

