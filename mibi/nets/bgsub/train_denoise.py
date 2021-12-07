import argparse
import logging
import sys

import torch

from mibi.nets.bgsub.dataset_patches import BackgroundSubtractionPatchesDataset
from mibi.nets.bgsub.network import BGSubAndDenoiser
from mibi.nets.bgsub.train import train_net


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

    def loss_func(chan_batch, bg_batch):
        transformed_batch = net(chan_batch, bg_batch)
        batch_energy = torch.sum(torch.pow(transformed_batch, 2))
        if batch_energy < 1e-9:
            msg = f"transformed batch energy is very low: {batch_energy}"
            raise RuntimeError(msg)
        return torch.log(torch.sum(transformed_batch * bg_batch)) - \
                torch.log(torch.sum(transformed_batch * chan_batch)) - \
                args.energy_coef*torch.log(batch_energy)

    try:

        train_net(net=net,
                  device=device,
                  dataset=ds,
                  loss_func=loss_func,
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
    