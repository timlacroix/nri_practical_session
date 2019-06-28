from tqdm import tqdm_notebook, tnrange
from torch.nn import BCEWithLogitsLoss
n_atoms = 5
model = MLPEncoder(n_atoms, int(4 * timesteps), 256, 2, 0.2).cuda()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def edge_accuracy(preds, target):
    """
    :param preds: edge logits
    :param target: ground truth
    :return: precision of the prediction
    """
    _, preds = preds.max(-1)
    correct = preds.float().data.eq(
        target.float().data.view_as(preds)).cpu().sum()
    return np.float(correct) / (target.size(0) * target.size(1))

# One epoch of training
def train(epoch):
    t = time.time()
    loss_train = []
    acc_train = []
    # pick the right loss
    loss = nn.BCEWithLogitsLoss(reduction='mean')
    model.train()
    with tqdm_notebook(loaders['train'], 'training') as t:
        for data, relations in t:
            data, relations = data.cuda(), relations.cuda()
            # compute the loss and take a gradient step
            logits = model(data)
            l = loss(
                logits,
                torch.cat((
                    (relations == 0)[:,:, None],
                    (relations == 1)[:,:, None]
                ), 2).float()
            )

            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            
            # compute the accuracy
            edge_acc = edge_accuracy(logits, relations)
            
            loss_train.append(l.item())
            acc_train.append(edge_acc)
            
            t.set_postfix(loss=l.item(), acc=edge_acc)

    return np.mean(loss_train), np.mean(acc_train)

# Train model
t_total = time.time()
best_epoch = 0
for epoch in tnrange(2):
    train_loss, train_acc = train(epoch)

