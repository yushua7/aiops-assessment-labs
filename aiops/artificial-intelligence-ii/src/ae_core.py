import numpy as np


def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total, n = 0.0, 0
    for (batch,) in loader:
        optimizer.zero_grad()
        recon = model(batch)
        loss = loss_fn(recon, batch)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(batch)
        n += len(batch)
    return total / n


def pick_threshold(train_errors, percentile):
    return float(np.percentile(train_errors, percentile))
