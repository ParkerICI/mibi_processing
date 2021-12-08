import os
import glob

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from ...preprocess.mibi_image import MIBIMultiplexImage


class BackgroundSubtractionRegressionDataset(Dataset):

    def __init__(self, images_dir, channels_data_path,
                 tif_subdir_name='TIFs', fov_prefix='Point',
                 gold_channel_label='197_Au'):

        # stack up all the images in a tensor of shape (nfovs, nchans, height, width)
        chan_indices = None
        X = list()
        df = {'fov':list()}
        for pdir in glob.glob(os.path.join(images_dir, f'{fov_prefix}*')):
            _,pname = os.path.split(pdir)
            print(f"Loading from {pdir}")
            tif_subdir = pdir
            if tif_subdir_name is not None:
                tif_subdir = os.path.join(tif_subdir, tif_subdir_name)
            mp_img = MIBIMultiplexImage.load_from_path(tif_subdir, channels_data_path, smooth=True)
            
            gold_chan_idx = mp_img.label_to_index[gold_channel_label]
            chan_indices = [gold_chan_idx] + list(mp_img.included_channel_indices())

            img = mp_img.X['raw'][chan_indices, :, :].astype('float32')
            X.append(torch.tensor(img))
            df['fov'].append(pname)
        
        X = torch.stack(X)

        # swap dimensions to shape (nfovs, height, width, nchans)
        self.X = X.permute([0, 2, 3, 1])

        # create mask
        df_channel = pd.read_csv(channels_data_path)
        self.M = create_mask(df_channel, chan_indices)

        self.chan_indices = chan_indices
        self.chan_names = [df_channel.loc[chan_idx]['Label'] for chan_idx in chan_indices]
        self.df = pd.DataFrame(df)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        assert index < self.X.shape[0], f"index={index}, but len(X)={len(self.X)}"
        return self.X[index]


def create_mask(df_channel, chan_indices):
    # create a mask for the regression matrix
    nchan = len(chan_indices)
    
    # 1's in column k of matrix M corresponds to
    # channels that contribute to the signal in row k of X
    M = np.zeros([nchan, nchan])
    mass_spec_tol = 0.1

    def between(x, low, high):
        if x >= low and x <= high:
            return True
        return False

    for k,chan_k in enumerate(chan_indices):
        mass_k = df_channel.loc[chan_k]['Isotope']
        for j,chan_j in enumerate(chan_indices):
            mass_j = df_channel.loc[chan_j]['Isotope']
            mass_diff = mass_j - mass_k
            if between(mass_diff, 1-mass_spec_tol, 1+mass_spec_tol):
                M[k, j] = 1
            if between(mass_diff, 16-mass_spec_tol, 16+mass_spec_tol):
                M[k, j] = 1
            M[0, j] = 1

    return torch.tensor(M.astype('float32'))