import argparse
import logging
import sys
import time
import json

import pandas as pd

import torch

from torch import optim
from torch.utils.data import DataLoader, random_split

from mibi.nets.bgsub.dataset_patches import BackgroundSubtractionPatchesDataset
from mibi.nets.bgsub.network import BGSubAndDenoiser

def train_net(net,
              device,
              dataset : BackgroundSubtractionPatchesDataset,
              epochs: int = 5,
              batch_size: int = 7,
              learning_rate: float = 0.001,
              weight_decay: float = 1,
              validation_fraction: float = 0.25,
              energy_coef = 1.,
              model_desc='default'):
    
    training_start_time = time.time()

    # 1. Split into train / validation partitions
    n_val = int(len(dataset) * validation_fraction)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    # 2. Create data loaders
    loader_args = dict(batch_size=batch_size)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    logging.info(f'''Starting training:
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

    def loss_func(chan_batch, bg_batch):
        transformed_batch = net(chan_batch, bg_batch)
        batch_energy = torch.sum(torch.pow(transformed_batch, 2))
        if batch_energy < 1e-9:
            msg = f"transformed batch energy is very low: {batch_energy}"
            raise RuntimeError(msg)
        return torch.log(torch.sum(transformed_batch * bg_batch)) - \
                torch.log(torch.sum(transformed_batch * chan_batch)) - \
                energy_coef*torch.log(batch_energy)

    # 4. Begin training
    df_loss = {'epoch':list(),
               'train_mean':list(), 'train_sd':list(),
               'validation_mean':list(), 'validation_sd':list()}

    for epoch_num in range(epochs):

        train_losses = list()
        for batch_num,(bg_batch,chan_batch) in enumerate(train_loader):

            start_time = time.time()

            net.train()
            if device.type != 'cpu':
                bg_batch = bg_batch.to(device=device, dtype=torch.float32)
                chan_batch = chan_batch.to(device=device, dtype=torch.float32)

            loss = loss_func(chan_batch, bg_batch)
            train_losses.append(loss)

            loss.backward()
            optimizer.step()

            elapsed_time = time.time() - start_time
            if batch_num % 20 == 0:
                print(f"Epoch {epoch_num}, batch {batch_num}: loss={loss:.6f}, time={elapsed_time:.2f}s")

        # run validation
        validation_losses = list()
        with torch.no_grad():
            start_time = time.time()

            net.eval()
            validation_losses = list()
            for batch_num,(bg_batch,chan_batch) in enumerate(val_loader):
                if device.type != 'cpu':
                    bg_batch = bg_batch.to(device=device, dtype=torch.float32)
                    chan_batch = chan_batch.to(device=device, dtype=torch.float32)    
                loss = loss_func(chan_batch, bg_batch)
                validation_losses.append(loss)    
            elapsed_time = time.time() - start_time

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
    all_params['energy_coef'] = energy_coef

    # save trained model
    out_fname = f"{model_desc}_network"
    torch.save(net.state_dict(), f"{out_fname}.pytorch.zip")
    with open(f"{model_desc}_params.json", 'w') as f:
        json.dump(all_params, f, indent=4)


def get_args():
    parser = argparse.ArgumentParser(description='Train the bgsub+denoising model on a set of MIBI images')
    parser.add_argument('--images_dir', type=str, help='Base directory where sets of images reside.',
                        required=True)
    parser.add_argument('--channel_file', type=str, help='Path to channel info file',
                        required=True)
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='Weight decay rate')
    parser.add_argument('--energy_coef', type=float, default=1,
                        help='Coefficient for energy regularization')
    parser.add_argument('--validation', type=float, default=0.1,
                        help='Fraction of the data that is used as validation (0-1)')
    parser.add_argument('--model_desc', type=str, default='default',
                        help='File pattern for model outputs.')

    return parser.parse_args()


def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    ds = BackgroundSubtractionPatchesDataset(args.images_dir, args.channel_file)
    net = BGSubAndDenoiser()

    if device.type != 'cpu':
        net.to(device=device)

    try:

        train_net(net=net,
                  device=device,
                  dataset=ds,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.learning_rate,
                  weight_decay=args.weight_decay,
                  validation_fraction=args.validation,
                  model_desc=args.model_desc,
                  energy_coef=args.energy_coef)

    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pytorch.zip')
        logging.info('Saved interrupt')
        sys.exit(0)


if __name__ == '__main__':
    args = get_args()
    main(args)
    