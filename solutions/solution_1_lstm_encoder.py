import torch
from torch import nn
from torch.functional import F

class RecurrentBaseline(nn.Module):
    """LSTM model for joint trajectory prediction."""

    def __init__(self, n_dims, n_hid, n_atoms, n_layers, do_prob=0.):
        super(RecurrentBaseline, self).__init__()
        
        # Encoder from positions to n_hid dimensional space
        # The architecture is linear / relu / dropout(do_prob) / linear / relu
        self.pos_encoder = nn.Sequential(
            nn.Linear(n_dims, n_hid),
            nn.ReLU(),
            nn.Dropout(p=do_prob),
            nn.Linear(n_hid, n_hid),
            nn.ReLU()
        )
        
        # RNN : n_atoms * n_hid -> n_atoms * n_hid. LSTM with n_layers.
        self.rnn = nn.LSTM(n_atoms * n_hid, n_atoms * n_hid, n_layers)  # TODO

        # Decode predicted *joint* configuration to physical *joint* location
        # The architecture is linear / relu / linear
        self.pos_decoder = nn.Sequential(
            nn.Linear(n_atoms * n_hid, n_atoms * n_hid),
            nn.ReLU(),
            nn.Linear(n_atoms * n_hid, n_atoms * n_dims)
        )

    def step(self, ins, hidden=None):
        # Input shape: [num_sims, n_atoms, n_dims]
        
        # Apply first MLP to encode the coordinates
        x = self.pos_encoder(ins)
        # View to join the last two dimensions. Add a dummy time dimension at the beginning of x
        x = x.view(ins.shape[0], -1).unsqueeze(0)
        # Apply LSTM given hidden and encoded input
        x, hidden = self.rnn(x, hidden)
        # remove extraneous time dimension
        x = x[0, :, :]
        # Apply second MLP to decode the output of the LSTM and compute delta
        delta = self.pos_decoder(x)
        # View to separate the last two dimensions again
        delta = delta.view(ins.size(0), ins.size(1), -1)
        # Add delta to inputs
        x = ins + delta

        # Return both output and hidden
        return x, hidden

    def forward(self, inputs, burn_in_steps=1):
        # Input shape: [num_sims, num_things, num_timesteps, n_in]

        outputs = []
        hidden = None

        for step in range(0, inputs.size(2) - 1):
            # If step <= burn_in_steps, the input is the true input
            # Otherwise it's the output of the previous step
            if step <= burn_in_steps:
                ins = inputs[:, :, step, :]
            else:
                ins = outputs[step - 1]

            output, hidden = self.step(ins, hidden)
            outputs.append(output)

        outputs = torch.stack(outputs, dim=2)

        return outputs

