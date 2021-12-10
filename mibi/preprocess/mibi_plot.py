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


def plot_img(img : np.ndarray, title='', transform=True, saturate_percentile=95):
    if transform:
        img_trans = clean_trans(np.arcsinh, img)
    else:
        img_trans = img
    
    vmax = img_trans.max()
    if saturate_percentile < 100:
        nz = img_trans > 0
        if nz.sum() > 0:
            vmax = np.percentile(img_trans[nz].ravel(), saturate_percentile)
    
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
                    transform=False, saturate_percentile=100,
                    transforms_to_show=['raw', 'hist', 'denoise'],
                    plot_height=4, plot_width=3, output_file=None):

    chan_indices = list()
    if exclude_ignored_channels:
        chan_indices.extend(mp_img.included_channel_indices())
    else:
        chan_indices = mp_img.df_channel.index.values

    ncols = len(transforms_to_show)
    nrows = len(chan_indices)
    fig_width = plot_width*ncols
    fig_height = plot_height*nrows
    
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
                plot_img(img, title=name, transform=transform, saturate_percentile=saturate_percentile)
                if col == 0:
                    plt.ylabel(side_title)
                col += 1
            plt.autoscale(tight=True)

    plt.tight_layout()

    if output_file is not None:
        fig.savefig(output_file)
    return fig


def default_plot_func(mp_img, transform, chan_label):
    gold_index = mp_img.label_to_index[mp_img.gold_channel_label]
    chan_idx = mp_img.label_to_index[chan_label]
    if chan_idx == gold_index:
        img = mp_img.X['raw'][chan_idx]
    else:
        img = mp_img.X[transform][chan_idx]

    plot_img(img, transform=False, saturate_percentile=99)


def transform_img(img, norm_percentile=99, thresh=0):
    cimg = img.copy()
    cimg[cimg < thresh] = 0.
    timg = np.arcsinh(cimg)
    p_norm = np.percentile(timg.ravel(), norm_percentile)
    if np.isnan(p_norm):
        p_norm = timg.max()
    timg /= p_norm
    timg[timg > 1] = 1
    return timg


def check_image(img, transform, chan_label):
    nz = img > 0
    if nz.sum() == 0:
        print(f"image is zero, transform={transform}, chan_label={chan_label}")
    na = np.isnan(img)
    if na.sum() > 0:
        print(f"image has nans, transform={transform}, chan_label={chan_label}")


def make_nuclear_merged_image(mp_img, transform, chan_label):
    
    nuclear_idx = mp_img.label_to_index['nuclear']
    chan_idx = mp_img.label_to_index[chan_label]

    nuclear_img = mp_img.X[transform][nuclear_idx].copy()
    chan_img = mp_img.X[transform][chan_idx].copy()

    img_shape = nuclear_img.shape
    final_img = np.zeros([img_shape[0], img_shape[1], 4])

    check_image(nuclear_img, transform, 'nuclear')
    nuclear_img = transform_img(nuclear_img, norm_percentile=100)
    nuclear_img *= 0.25

    check_image(chan_img, transform, chan_label)
    chan_img = transform_img(chan_img, norm_percentile=100, thresh=0.0)

    final_img[:, :, 0] = chan_img
    final_img[:, :, 1] = nuclear_img
    final_img[:, :, 2] = chan_img
    final_img[:, :, 3] = 0.95

    return final_img


def plot_merged_channels(mp_img, transform, chan_label):
    img = make_nuclear_merged_image(mp_img, transform, chan_label)
    plt.imshow(img, interpolation='nearest', aspect='auto')
    ax = plt.gca()
    ax.set_xticks([], [])
    ax.set_yticks([], [])


def plot_image_grid(mp_images, channels=None, transform='bgsub',
                   plot_func=default_plot_func, fov_order=None, fov_labels=dict(),
                   out_file=None, transpose=False):
    """ Plots a grid where each column is a FOV and each row is a channel. """
    
    if fov_order is None:
        fov_order = mp_images.keys()
     
    plot_height = 8
    plot_width = 6.5

    ncols = len(fov_order)
    if channels is None:
        mp_img = mp_images[fov_order[0]]
        nrows = len(mp_img.df_channel)
    else:
        nrows = len(channels)

    if transpose:
        ncols_copy = ncols
        ncols = nrows
        nrows = ncols_copy

    fig_width = plot_height*ncols
    fig_height = plot_width*nrows

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = plt.GridSpec(nrows, ncols)

    for col_idx,fov in enumerate(fov_order):
        mp_img = mp_images[fov]
        gold_index = mp_img.label_to_index[mp_img.gold_channel_label]
        
        if channels is None:
            channel_indices = [gold_index] + list(mp_img.included_channel_indices())
        else:
            channel_indices = [mp_img.label_to_index[c] for c in channels]

        if fov in fov_labels:
            col_title = fov_labels[fov]
        else:
            col_title = fov

        for row_idx,chan_idx in enumerate(channel_indices):
            chan_label = mp_img.df_channel.loc[chan_idx]['Label']

            if transpose:
                ax = fig.add_subplot(gs[col_idx, row_idx])
            else:
                ax = fig.add_subplot(gs[row_idx, col_idx])
                
            plot_func(mp_img, transform, chan_label)

            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
            plt.ylabel(chan_label, fontsize=25)

            if transpose:    
                plt.title(col_title, fontsize=25, loc='left')
            else:
                if row_idx == 0:
                    plt.title(col_title, fontsize=20)
                if row_idx == nrows-1:
                    plt.title(col_title, y=-0.15, fontsize=20)

    plt.tight_layout()
    if out_file is not None:
        fig.savefig(out_file)
