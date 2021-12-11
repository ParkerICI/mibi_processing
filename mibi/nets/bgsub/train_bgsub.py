import argparse
import logging
import sys

import numpy as np

import torch
import torch.nn as nn

from mibi.nets.bgsub.dataset_regression import BackgroundSubtractionRegressionDataset
from mibi.nets.bgsub.network import BGSubtractRegression
from mibi.nets.bgsub.train import train_net


def get_args():
    parser = argparse.ArgumentParser(description='Train the bgsub+denoising model on a set of MIBI images')
    parser.add_argument('--images_dir', type=str, help='Base directory where sets of images reside.',
                        required=True)
    parser.add_argument('--channel_file', type=str, help='Path to channel info file',
                        required=True)
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-2,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='Weight decay rate')
    parser.add_argument('--validation', type=float, default=0.1,
                        help='Fraction of the data that is used as validation (0-1)')
    parser.add_argument('--model_desc', type=str, default='default',
                        help='File pattern for model outputs.')

    return parser.parse_args()


def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    ds = BackgroundSubtractionRegressionDataset(args.images_dir, args.channel_file)
    net = BGSubtractRegression(ds.M)

    if device.type != 'cpu':
        net.to(device=device)

    mse_loss = nn.MSELoss()
    def loss_func(img):
        pred_img = net(img)
        return mse_loss(pred_img, img)

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
                  model_desc=args.model_desc)

    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pytorch.zip')
        logging.info('Saved interrupt')
        sys.exit(0)

    # load trained model
    param_file = f'{args.model_desc}_network.pytorch.zip'
    net = BGSubtractRegression.load_from_file(param_file, ds.M)

    # write compensation matrix to file
    W = net.get_comp_matrix()
    np.savetxt('comp_matrix.csv', W, delimiter=',', fmt='%0.6f')
    with open('comp_matrix_channels.csv', 'w') as f:
        f.write(','.join(ds.chan_names))

if __name__ == '__main__':
    args = get_args()
    main(args)
    