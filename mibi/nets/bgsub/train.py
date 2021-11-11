import argparse
import logging
import sys

import torch
from torch import optim
from torch.utils.data import DataLoader, random_split

from pytorch_msssim import MS_SSIM

from mibi.nets.bgsub.dataset import BackgroundSubtractionDataset
from mibi.nets.bgsub.network import BGSubAndDenoisier

DEFAULT_LOSS_PARAMS = {
                       "size_average": True,
                       "win_size": 11,
                       "win_sigma": 1.5,
                       "channel": 1,
                       "spatial_dims": 2,
                       "weights": None,
                       "K": (0.01, 0.03)
                      }

def train_net(net,
              device,
              dataset : BackgroundSubtractionDataset,
              epochs: int = 5,
              batch_size: int = 7,
              learning_rate: float = 0.001,
              weight_decay: float = 1,
              validation_fraction: float = 0.1,
              loss_params: dict = DEFAULT_LOSS_PARAMS):
    
    # 1. Split into train / validation partitions
    n_val = int(len(dataset) * validation_fraction)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 2. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
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
    ms_ssim_loss = MS_SSIM(data_range=dataset.range[1], **loss_params)

    optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 4. Begin training
    all_losses = list()
    for epoch_num in range(epochs):
        epoch_losses = list()

        for batch_num,(bg_batch,chan_batch) in enumerate(train_loader):
            bg_batch = bg_batch.to(device=device, dtype=torch.float32)
            chan_batch = chan_batch.to(device=device, dtype=torch.float32)

            transformed_batch = net(chan_batch, bg_batch)

            bg_sim = ms_ssim_loss(transformed_batch, bg_batch)
            chan_sim = ms_ssim_loss(transformed_batch, chan_batch)

            loss = 0.5*bg_sim - 0.5*chan_sim

            epoch_losses.append(loss)

            if batch_num % 5 == 0:
                print(f"Epoch {epoch_num}, batch {batch_num}: loss={loss:.6f}")

            loss.backward()
            optimizer.step()

        all_losses.append(epoch_losses)

    # save trained model
    torch.save(net.state_dict(), "bgsub+noise_net.dict.zip")


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
    parser.add_argument('--validation', type=float, default=0.1,
                        help='Fraction of the data that is used as validation (0-1)')
    parser.add_argument('--loss_win_size', type=float, default=11,
                        help='Window size for MS_SSIM loss')
    parser.add_argument('--loss_win_sigma', type=float, default=1.5,
                        help='Window size for MS_SSIM loss')
    parser.add_argument('--loss_K1', type=float, default=0.01,
                        help='K1 param for MS_SSIM loss')
    parser.add_argument('--loss_K2', type=float, default=0.03,
                        help='K2 param for MS_SSIM loss')


    return parser.parse_args()


def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    ds = BackgroundSubtractionDataset(args.images_dir, args.channel_file)
    net = BGSubAndDenoisier()

    net.to(device=device)

    loss_params = DEFAULT_LOSS_PARAMS
    loss_params['win_size'] = args.loss_win_size
    loss_params['win_sigma'] = args.loss_win_sigma
    loss_params['K'] = (args.loss_K1, args.loss_K2)

    try:

        train_net(net=net,
                  device=device,
                  dataset=ds,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.learning_rate,
                  weight_decay=args.weight_decay,
                  validation_fraction=args.validation,
                  loss_params=loss_params)

    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)


if __name__ == '__main__':
    args = get_args()
    main(args)
    