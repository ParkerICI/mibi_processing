import argparse
import logging
import sys
import time
import json

import pandas as pd

import torch

from torch import optim
from torch.utils.data import DataLoader, random_split


def is_multi(x):
    return isinstance(x, (list, tuple))


def to_device(batch, device):
    if not is_multi(batch):
        batch = [batch]
    if device.type != 'cpu':
        batch = [b.to(device=device, dtype=torch.float32) for b in batch]
    return batch


def train_net(net,
              device,
              dataset,
              loss_func,
              epochs: int = 5,
              batch_size: int = 7,
              learning_rate: float = 0.001,
              weight_decay: float = 1,
              validation_fraction: float = 0.25,
              hyperparams: dict = dict(),
              model_desc='default'):
    
    training_start_time = time.time()

    # 1. Split into train / validation partitions
    n_val = int(len(dataset) * validation_fraction)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    # 2. Create data loaders
    loader_args = dict(batch_size=batch_size)
    train_loader = DataLoader(train_set, shuffle=False, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    print(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Decay rate:      {weight_decay}
        Training size:   {n_train}
        Validation size: {n_val}
        Device:          {device.type}
    ''')

    # 3. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 4. Begin training
    df_loss = {'epoch':list(),
               'train_mean':list(), 'train_sd':list(),
               'validation_mean':list(), 'validation_sd':list()}

    for epoch_num in range(epochs):

        train_losses = list()
        for batch_num,batch in enumerate(train_loader):
            start_time = time.time()

            net.train()
            batch = to_device(batch, device)
            
            loss = loss_func(*batch)
            train_losses.append(loss)

            loss.backward()
            optimizer.step()

            elapsed_time = time.time() - start_time
            if batch_num % 20 == 0:
                print(f"Epoch {epoch_num}, batch {batch_num}: loss={loss:.6f}, time={elapsed_time:.2f}s")

        # run validation
        if n_val > 0:
            validation_losses = list()
            with torch.no_grad():
                start_time = time.time()

                net.eval()
                validation_losses = list()
                for batch_num,batch in enumerate(val_loader):
                    batch = to_device(batch, device)
                    loss = loss_func(*batch)
                    validation_losses.append(loss)
                elapsed_time = time.time() - start_time
        else:
            validation_losses = [0.]

        train_loss_mean = torch.mean(torch.tensor(train_losses))
        train_loss_sd = torch.std(torch.tensor(train_losses))        
        validation_loss_mean = torch.mean(torch.tensor(validation_losses))
        validation_loss_sd = torch.std(torch.tensor(validation_losses))

        df_loss['epoch'].append(epoch_num)
        df_loss['train_mean'].append(train_loss_mean.detach().numpy())
        df_loss['train_sd'].append(train_loss_sd.detach().numpy())
        df_loss['validation_mean'].append(validation_loss_mean.detach().numpy())
        df_loss['validation_sd'].append(validation_loss_sd.detach().numpy())

        print(f"""----- Epoch {epoch_num} ------
        training loss={train_loss_mean:.6f} +/- {train_loss_sd:.6f}
        validation loss={validation_loss_mean:.6f} +/- {validation_loss_sd:.6f}
        elapsed time={elapsed_time:.2f}s
        """
        )

    training_elapsed_time = time.time() - training_start_time

    # save losses
    df_loss = pd.DataFrame(df_loss)
    df_loss.to_csv(f"{model_desc}_losses.csv", header=True, index=False)

    # save parameters
    all_params = dict()
    all_params['learning_rate'] = learning_rate
    all_params['weight_decay'] = weight_decay
    all_params['epochs'] = epochs
    all_params['batch_size'] = batch_size
    all_params['validation_loss_mean'] = float(validation_loss_mean.detach().numpy())
    all_params['validation_loss_sd'] = float(validation_loss_sd.detach().numpy())
    all_params['training_elapsed_time'] = training_elapsed_time
    all_params['model_desc'] = model_desc
    for hname,hval in hyperparams.items():
        all_params[hname] = hval

    # save trained model
    out_fname = f"{model_desc}_network"
    torch.save(net.state_dict(), f"{out_fname}.pytorch.zip")
    with open(f"{model_desc}_params.json", 'w') as f:
        json.dump(all_params, f, indent=4)

