import argparse
import logging
import sys
import time
import json

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch import optim

from mibi.nets.bgsub.network import BGSubtractRegression

def train_net(net,
              device,
              X,
              epochs: int = 5,
              learning_rate: float = 0.001,
              weight_decay: float = 1,
              model_desc='default'):
    
    training_start_time = time.time()

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Learning rate:   {learning_rate}
        Decay rate:      {weight_decay}
        Device:          {device.type}
    ''')

    # 3. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    mse_loss = nn.MSELoss()
    def loss_func(img):
        pred_img = net(img)
        return mse_loss(pred_img, img)
        
    # 4. Begin training
    df_loss = {'epoch':list(), 'train_err':list()}

    if device.type != 'cpu':            
        X = X.to(device=device, dtype=torch.float32)

    for epoch_num in range(epochs):
        start_time = time.time()

        net.train()
        
        loss = loss_func(X)
        
        loss.backward()
        optimizer.step()

        elapsed_time = time.time() - start_time
        
        df_loss['epoch'].append(epoch_num)
        df_loss['train_err'].append(loss.detach().numpy())
        
        print(f"""----- Epoch {epoch_num} ------
        training loss={loss:.6f}
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
    all_params['training_elapsed_time'] = training_elapsed_time
    all_params['model_desc'] = model_desc
    
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

    net = BGSubtractRegression()

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
    