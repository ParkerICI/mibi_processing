from typing import Sequence

import torch

from mibi.nets.bgsub.dataset import BackgroundSubtractionDataset
from mibi.nets.bgsub.network import BGSubAndDenoiser

def transform_fovs(ds : BackgroundSubtractionDataset,
                   fov_names : Sequence[str],
                   net : BGSubAndDenoiser) -> dict:

    i = ds.df['fov_name'].isin(fov_names)
    df = ds.df[i]

    images = dict()
    
    for fov_name, sub_df in df.groupby('fov_name'):
        transformed_channels = dict()
        for chan_name,sub_sub_df in sub_df.groupby('channel_label'):
            #print(f"{fov_name} / {chan_name}")
            trans_patches = list()
            for ds_index in sub_sub_df.index:
                gold_patches, chan_patches = ds[ds_index]
                Y = net(chan_patches.unsqueeze(0), gold_patches.unsqueeze(0))
                trans_patches.append(Y)
            trans_patches = torch.stack(trans_patches).squeeze()
            trans_patches = trans_patches.permute([1, 2, 0])
            #print('trans_patches.shape=',trans_patches.shape)
            trans_img = ds.pm.unpatch(trans_patches.unsqueeze(0).unsqueeze(0))
            #print('trans_img.shape=',trans_img.shape)
            transformed_channels[chan_name] = trans_img.squeeze()

        images[fov_name] = transformed_channels
    
    return images
