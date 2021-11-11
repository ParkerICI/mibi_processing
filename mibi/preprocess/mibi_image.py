import os
import argparse

from tifffile import TiffFile

import numpy as np
import pandas as pd

from scipy.ndimage import gaussian_filter

from sklearn.linear_model import Ridge

from mibi.preprocess.knn_denoise import knn_denoise

LABEL_DESC = {'12_C':'Carbon',
              '23_Na':'Sodium',
              '28_Si':'Silicon',
              '31_P':'Phosphorous',
              '38_calib':'Calibration',
              '40_Ca':'Calcium',
              '56_Fe':'Iron',
              'nuclear':'Nuclear',
              'B7H3':'B7H3 T-cell Checkpoint',
              'EGFR':'Growth Factor Receptor',
              'IL13RA2':'Tumor Associated',
              'NeuN':'Neuron',
              'Epha2':'Tumor Marker?',
              'CD3':'T-cell Marker',
              'MMP2':'ECM Degredation',
              'H3K27M':'Diffuse Glioma Marker',
              'CD86':'B7 Immune Co-stimulation',
              'CD14':'Macrophage Marker',
              'CD45':'Immune Marker',
              'CD123':'IL3 Receptor',
              'GFAP':'Astrocyte Marker',
              '181_Ta':'Tantalum',
              '197_Au':'Gold'
            }


class MIBIMultiplexImage(object):

    def __init__(self):
        self.X = None
        self.X_bgsub = dict()
        self.X_thresh = dict()
        self.X_denoise = dict()
        
        self.df_channel = None
    
        self.gold_channel_label = '197_Au'
        self.ignored_channel_labels = ['12_C', '23_Na', '28_Si', '31_P', '38_calib', '40_Ca', '56_Fe', '181_Ta']
        self.label_to_index = dict()
    
    @staticmethod
    def load_from_path(img_dir, channels_data_path, smooth=True):
        mp_img = MIBIMultiplexImage()
        
        mp_img.df_channel = pd.read_csv(channels_data_path)
        mp_img.df_channel['Description'] = mp_img.df_channel['Label'].map(lambda x: LABEL_DESC[x])

        mp_img.label_to_index = {row['Label']:row_idx for row_idx,row in mp_img.df_channel.iterrows()}

        images = list()
        for lbl in mp_img.df_channel['Label']:
            tiff_file = os.path.join(img_dir, f'{lbl}.tif')
            if not os.path.exists(tiff_file):
                print('Missing channel: {tiff_file}')
                img = np.zeros([1024, 1024])*np.nan
            else:
                tf = TiffFile(tiff_file)
                img = tf.asarray()

            if smooth:
                img = gaussian_filter(img.astype('float32'), sigma=1)

            images.append(img)

        mp_img.X = np.array(images)
    
        return mp_img

    def included_channel_indices(self):
        """ Get indices of channels that are to be processed and displayed (except gold channel).
            The index of an element in self.df_channel matches the index of an image in self.X
        """
        channels_to_include = np.setdiff1d(self.df_channel['Label'],
                                           self.ignored_channel_labels + [self.gold_channel_label])
        i = self.df_channel['Label'].isin(channels_to_include)
        return self.df_channel.index[i].values
    
    def bg_subtract(self, debug=False):

        # identify channels that would add noise
        channel_dependencies = dict()
        channels_to_include = self.included_channel_indices()
        gold_channel_index = self.label_to_index[self.gold_channel_label]
        
        df_deps = self.df_channel.loc[list(channels_to_include) + [gold_channel_index]]

        for chan_idx in channels_to_include:
            
            deps = [gold_channel_index]
            chan_info = self.df_channel.loc[chan_idx]

            # get -1 channel index
            baseline_start = chan_info['BaselineStart']
            bdiff = (baseline_start - df_deps['BaselineStart'])
            i = (bdiff < 2) & (bdiff > 0)
            if i.sum() > 0:
                deps.extend(df_deps.loc[i].index)
            
            # get -16 channel index
            i = (bdiff > 15) & (bdiff < 17)
            if i.sum() > 0:
                deps.extend(df_deps.loc[i].index)
            
            channel_dependencies[chan_idx] = deps

        # get coefficients using ridge regression for each image
        coefs = dict()
        for chan_idx,deps in channel_dependencies.items():
            chan_label = self.df_channel.loc[chan_idx]['Label']
            dep_names = ', '.join([self.df_channel.loc[k]['Label'] for k in deps])
            if debug:
                print(f'Regressing on {chan_label}, deps={dep_names}')
            y = self.X[chan_idx, :, :].ravel()
            R = self.X[deps, :, :].reshape([len(deps), -1]).T

            rr = Ridge(fit_intercept=False, positive=True)
            rr.fit(R, y)
            coefs[chan_idx] = rr.coef_
        self.bgsub_coefs = coefs

        # perform background subtraction using regression coefficients 
        self.X_bgsub = dict()
        for chan_idx, weights in coefs.items():
            img = self.X[chan_idx, :, :].copy()
            deps = channel_dependencies[chan_idx]
            for dep_chan_idx,w in zip(deps, weights):
                S = self.X[dep_chan_idx, :, :]
                img -= w*S
            img[img < 0] = 0
            self.X_bgsub[chan_idx] = img

    def threshold(self, percentile_threshold=50, debug=False):
        """ Threshold out all pixels below the given percentile of the channel's intensity histogram. """
        self.X_thresh = dict()

        for chan_idx,img in self.X_bgsub.items():
            nz = img > 0
            thresh = np.percentile(img[nz].ravel(), percentile_threshold)
            if debug:
                print(f"{self.df_channel.loc[chan_idx]['Label']} q{percentile_threshold}={thresh:0.3f}")
            q_img = img.copy() 
            q_img[q_img < thresh] = 0.
            self.X_thresh[chan_idx] = q_img

    def denoise(self):
        self.X_denoise = dict()
        for chan_idx,img in self.X_thresh.items():
            self.X_denoise[chan_idx] = knn_denoise(img)

    def preprocess(self, debug=False):
        self.bg_subtract(debug=debug)
        self.threshold(debug=debug)
        self.denoise()


def main(args):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run FastICA on a set of .tiff files in a directory.')
    parser.add_argument('--input_dir', type=str, help="A directory that contains tiff files.")
    parser.add_argument('--output_dir', type=str, help="Directory to write output tiff files to")
    parser.add_argument('--smooth', type=bool, help="Smooth inputs with 1px Gaussian before ICA", default=False)
    parser.add_argument('--files', default=None,
                        help='Comma-separated list of filenames within input_dir to process.')

    args = parser.parse_args()

    main(args)
