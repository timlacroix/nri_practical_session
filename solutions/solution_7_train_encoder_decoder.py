from torch.nn.functional import gumbel_softmax, softmax

def kl_categorical_uniform(preds, num_atoms):
    kl_div = preds * torch.log(preds + 1e-16)
    return kl_div.sum() / (num_atoms * preds.size(0))

def train(data_loader, optimizer, encoder, decoder):
    loss_train = []

    encoder.train()
    decoder.train()
    with tqdm_notebook(data_loader, desc=f'training') as t:
        for data, relations in t:
            data, relations = data.cuda(), relations.cuda()     

            # Encode
            logits = encoder(data)

            # Compute edges with soft gumbel_softmax
            edges = gumbel_softmax(
                logits.view(-1, 2), tau=temp, hard=False
            ).view(logits.shape)

            # Decode using the edge weights
            output = decoder(
                data, edges, burn_in_steps=timesteps-prediction_steps
            )

            nll = nll_gaussian(
                output, data[:,:,1:,:], var
            )
            kl = kl_categorical_uniform(
                softmax(logits, 2), output.shape[1]
            )
            l = nll + kl
            
            edge_acc = edge_accuracy(logits, relations)

            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            loss_train.append(l.item())
            t.set_postfix(loss=l.item(), nll = nll.item(), kl = kl.item(), acc=edge_acc)
        
    return np.mean(loss_train)

dropout = 0
n_dims = 4
hidden = 256

encoder = MLPEncoder(num_atoms, int(n_dims * timesteps), hidden, 2, dropout).cuda()
decoder = RNNDecoder(n_dims, hidden, dropout).cuda()

optimizer = optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()),
    lr=learning_rate
)

for e in range(10):
    loss = train(loaders['train'], optimizer, encoder, decoder)
    print(f"{loss}")
