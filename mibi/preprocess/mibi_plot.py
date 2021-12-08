import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from mibi.preprocess.mibi_image import MIBIMultiplexImage

def clean_trans(trans_func, x):
    y = trans_func(x)
    i = np.isnan(y) | np.isinf(y)
    y[i] = 0.
    y[y < 0] = 0.
    return y


def plot_hist(img : np.ndarray, ax=None, exclude_zeros=True, channel_label=''):
    if ax is None:
        ax = plt.gca()
    i = img > 0
    zero_frac = (~i).sum() / np.prod(img.shape)
    if exclude_zeros:
        x = img[i].ravel()
    else:
        x = img.ravel()
    sns.histplot(x, bins=25, ax=ax)
    plt.title(f"{channel_label}; zfrac={zero_frac:.2f}")
    ax.set_yscale('log')
    plt.autoscale(tight=True)


def plot_img(img : np.ndarray, title='', transform=True, saturate=True):
    if transform:
        img_trans = clean_trans(np.arcsinh, img)
    else:
        img_trans = img
    
    if saturate:
        nz = img_trans > 0
        if nz.sum() > 0:
            vmax = np.percentile(img_trans[nz].ravel(), 95)
    else:
        vmax = img_trans.max()
    nz = img_trans > 0
    
    plt.imshow(img_trans, interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot, vmin=0, vmax=vmax)
    plt.xlabel('')
    plt.ylabel('')
    plt.xticks([]) 
    plt.yticks([])
    plt.colorbar()
    plt.title(title)
    plt.autoscale(tight=True)


def plot_intensity_vs_area(df_region, ax, area_thresh=3):
    i = df_region['convex_area'] > area_thresh
    if i.sum() > 0:
        try:
            sns.kdeplot(data=df_region[i], x='mean_intensity', y='convex_area', ax=ax, fill=True)
            sns.scatterplot(x='mean_intensity', y='convex_area', data=df_region[i], ax=ax, alpha=0.1, color='k')                         
            plt.yscale('log')
        except:
            print(f"Error with kdeplot, i.sum()={i.sum()}")


def plot_mibi_image(mp_img : MIBIMultiplexImage, exclude_ignored_channels=True,
                    transform=False, saturate=False,
                    transforms_to_show=['raw', 'hist', 'denoise'],
                    plot_height=4, plot_width=3, output_file=None):

    chan_indices = list()
    if exclude_ignored_channels:
        chan_indices.extend(mp_img.included_channel_indices())
    else:
        chan_indices = mp_img.df_channel.index.values

    ncols = len(transforms_to_show)
    nrows = len(chan_indices)
    fig_width = plot_height*ncols
    fig_height = plot_width*nrows
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = plt.GridSpec(nrows, ncols)

    for row,chan_idx in enumerate(chan_indices):
        chan_label = mp_img.df_channel.loc[chan_idx]['Label']
        chan_desc = mp_img.df_channel.loc[chan_idx]['Description']
        side_title = f"{chan_label}: {chan_desc}"

        col = 0
        for name in transforms_to_show:
            if name == 'hist':
                ax = fig.add_subplot(gs[row, col])
                plot_hist(mp_img.X['raw'][chan_idx, :, :], exclude_zeros=True, ax=ax)
                if col == 0:
                    plt.ylabel(side_title)
                col += 1
            else:
                ax = fig.add_subplot(gs[row, col])
                if isinstance(mp_img.X[name], dict):
                    img = mp_img.X[name][chan_idx]
                else:
                    img = mp_img.X[name][chan_idx, :, :]
                plot_img(img, title=name, transform=transform, saturate=saturate)
                if col == 0:
                    plt.ylabel(side_title)
                col += 1
            plt.autoscale(tight=True)

    plt.tight_layout()

    if output_file is not None:
        fig.savefig(output_file)
    return fig
