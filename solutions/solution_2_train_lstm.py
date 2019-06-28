from tqdm import tqdm_notebook, tnrange

model = RecurrentBaseline(4, 256, 5, 2, 0.2).cuda()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def nll_gaussian(preds, target, variance):
    neg_log_p = ((preds - target) ** 2 / (2 * variance))
    return neg_log_p.sum() / (target.size(0) * target.size(1))

# One epoch of training
def train(epoch):
    t = time.time()
    loss = []
    pred_loss = []
    model.train()
    with tqdm_notebook(loaders['train'], desc=f'training') as t:
        for data, relations in t:
            data, relations = data.cuda(), relations.cuda()
            # compute the predicted trajectory with burn_in = timesteps - prediction_steps
            output = model(data, burn_in_steps=timesteps-prediction_steps)

            # output_t is data_{t+1}. Select a time-shifted slice of target to make loss computations easier
            target = data[:, :, 1:, :]
            
            # Compute the training loss and nll on steps after burn in
            l = nll_gaussian(output, target, var)
            predicted_loss = nll_gaussian(output[:, :, -prediction_steps:, :], target[:, :, -prediction_steps:, :], var)

            # take a gradient step
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            loss.append(l.item())
            pred_loss.append(predicted_loss.item())
            
            t.set_postfix(loss=l.item(), pred=predicted_loss.item())

    return np.mean(loss), np.mean(pred_loss) 

# Train model
for epoch in tnrange(1):
    nll, pred = train(epoch)

