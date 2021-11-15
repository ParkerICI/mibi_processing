import os
import glob

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from ...preprocess.mibi_image import MIBIMultiplexImage
from ..utils import PatchMaker


def patch_and_swap(patch_maker, img):
    """ Take an image of shape (height, width), patchify it, and return an image of shape (npatches, height, width). """
    assert len(img.shape) == 2
    img_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0)
    img_patches = patch_maker.patch(img_tensor).squeeze()
    return img_patches.permute([2, 0, 1])


class BackgroundSubtractionDataset(Dataset):

    def __init__(self, images_dir, channels_data_path, patch_size=512, image_size=(1024, 1024),
                 patch_stride=256,
                 tif_subdir_name='TIFs', fov_prefix='Point',
                 gold_channel_label='197_Au'):
        
        self.gold_patches = dict()
        self.chan_patches = dict()
        self.range = [0, 0]

        self.pm = PatchMaker(image_size=image_size, kernel_size=patch_size, stride=patch_stride)

        # create dataframe used for indexing
        self.df = {'fov_name':list(), 'channel_label':list(), 'channel_local_index':list(),
                   'patch_index':list()}

        for pdir in glob.glob(os.path.join(images_dir, f'{fov_prefix}*')):
            _,pname = os.path.split(pdir)
            print(f"Loading from {pdir}")
            tif_subdir = pdir
            if tif_subdir_name is not None:
                tif_subdir = os.path.join(tif_subdir, tif_subdir_name)
            mp_img = MIBIMultiplexImage.load_from_path(tif_subdir, channels_data_path, smooth=True)

            # get gold channel image, patchify
            gold_chan_idx = mp_img.label_to_index[gold_channel_label]
            self.gold_patches[pname] = patch_and_swap(self.pm, mp_img.X['raw'][gold_chan_idx, :, :])

            self.range[0] = min(self.range[0],
                                self.gold_patches[pname].min())

            self.range[1] = max(self.range[1],
                                self.gold_patches[pname].max())
            
            # get other channel images, patchify
            patches = list()
            for chan_idx in mp_img.included_channel_indices():
                patches.append(patch_and_swap(self.pm, mp_img.X['raw'][chan_idx, :, :]))
            self.chan_patches[pname] = torch.stack(patches)

            self.range[0] = min(self.range[0],
                                self.chan_patches[pname].min())

            self.range[1] = max(self.range[1],
                                self.chan_patches[pname].max())

            # fill up the data frame with info
            num_patches = self.chan_patches[pname].shape[1]
            idx2label = {v:k for k,v in mp_img.label_to_index.items()}
            for k,chan_idx in enumerate(mp_img.included_channel_indices()):
                for j in range(num_patches):
                    self.df['fov_name'].append(pname)
                    self.df['channel_label'].append(idx2label[chan_idx])
                    self.df['channel_local_index'].append(k)
                    self.df['patch_index'].append(j)

        self.df = pd.DataFrame(self.df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if index > len(self.df)-1:
            raise IndexError(f'index={index}, number of elements is {len(self.df)}')

        row = self.df.loc[index]

        return (self.gold_patches[row['fov_name']][row['patch_index']].unsqueeze(0),
                self.chan_patches[row['fov_name']][row['channel_local_index']][row['patch_index']].unsqueeze(0))
