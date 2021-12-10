import os

from tifffile import TiffFile, imsave

import numpy as np
import pandas as pd

from scipy.ndimage import gaussian_filter

from sklearn.linear_model import Ridge

from skimage.filters import farid_h, farid_v

from mibi.preprocess.knn_denoise import knn_denoise

LABEL_DESC = {'12_C':'Carbon',
              '23_Na':'Sodium',
              '28_Si':'Silicon',
              '31_P':'Phosphorous',
              '38_calib':'Calibration',
              '40_Ca':'Calcium',
              '56_Fe':'Iron',
              'nuclear':'Nuclear',
              'Nuclear':'Nuclear',
              'SMA':'Unknown',
              'LAG3':'T-cell Inhibitor',
              'CD4':'Helper T-cells',
              'PD1':'Immune Checkpoint',
              'PDL1':'Ligand for PD-1 receptor',
              'CD56':'NCAM (neurons, glia, muscle)',
              'Gran_B':'Granzyme B in T-cells',
              'CD68':'Monocytes, macrophages, microglia',
              'CD11c':'Dendritic Cells',
              'CD3e':'CD3 Co-receptor (T-cells)',
              'IDO':'Immune Checkpoint',
              'ICOS':'Co-stimulatory receptor on T-cells',
              'FoxP3':'Treg Marker',
              'CD31':'Immune and Endothelial',
              'CD8':'Effector T-cells',
              'CD3':'T-cell Marker',
              'CD163':'Monocytes/Macrophages',
              'TMEM119':'Activated Microglia',
              'CD208':'Dendritic Cells',
              'HLA_DR':'MHC II',
              'CD133':'Stem Cell Marker',
              'CD209':'Macrophages/DCs',
              'Ki67':'Proliferation Marker',
              'HLA1':'MHC I',
              'Vimentin':'Mesenchymal Cell marker',
              'CD20':'B-cells',
              'B7H3':'B7H3 T-cell Checkpoint',
              'EGFR':'Growth Factor Receptor',
              'Calprotectin':'Neutrophils/Inflammation',
              'Chymase_Tryptase':'Mast Cells',
              'IL13RA2':'Tumor Associated',
              'PanCK':'Cytokeratin/Epithelial',
              'NeuN':'Neuron',
              'Epha2':'Tumor Marker?',
              'CD3':'T-cell Marker',
              'MMP2':'ECM Degredation',
              'H3K27M':'Diffuse Glioma Marker',
              'CD86':'B7 Immune Co-stimulation',
              'TIM3':'T-cell Exhaustion',
              'CD14':'Macrophage Marker',
              'CD45':'Immune Marker',
              'CD45RO':'Memory T-cells',
              'CD123':'IL3 Receptor',
              'GFAP':'Astrocyte Marker',
              '181_Ta':'Tantalum',
              '197_Au':'Gold'
            }

IMAGE_STATS_PERCENTILES = (1, 10, 25, 50, 75, 90, 99)

def image_stats(img, percentiles=IMAGE_STATS_PERCENTILES):
    d = dict()
    i = img > 0
    d['intensity_sparsity'] = (~i).sum() / np.prod(img.shape)
    
    d['percentiles'] = percentiles
    d['intensity_percentiles'] = np.percentile(img[i].ravel(), percentiles)
    
    p99 = np.percentile(img.ravel(), 99)
    img_norm = img / p99
    img_norm[img_norm > 1] = 0
    grad_img = np.concatenate([farid_h(img_norm), farid_v(img_norm)])
    i = grad_img > 0
    d['grad_sparsity'] = (~i).sum() / np.prod(grad_img.shape)
    d['grad_percentiles'] = np.percentile(grad_img[i].ravel(), percentiles)
    
    return d


class MIBIMultiplexImage(object):

    def __init__(self):
        self.X = dict()

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
                print(f'Missing channel: {tiff_file}')
                img = np.zeros([1024, 1024])*np.nan
            else:
                tf = TiffFile(tiff_file)
                img = tf.asarray()

            if smooth:
                img = gaussian_filter(img.astype('float32'), sigma=1)

            images.append(img)

        mp_img.X['raw'] = np.array(images)
    
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
            y = self.X['raw'][chan_idx, :, :].ravel()
            R = self.X['raw'][deps, :, :].reshape([len(deps), -1]).T

            rr = Ridge(fit_intercept=False, positive=True)
            rr.fit(R, y)
            coefs[chan_idx] = rr.coef_
        self.bgsub_coefs = coefs

        # perform background subtraction using regression coefficients 
        self.X['bgsub'] = dict()
        for chan_idx, weights in coefs.items():
            img = self.X['raw'][chan_idx, :, :].copy()
            deps = channel_dependencies[chan_idx]
            for dep_chan_idx,w in zip(deps, weights):
                S = self.X['raw'][dep_chan_idx, :, :]
                img -= w*S
            img[img < 0] = 0
            self.X['bgsub'][chan_idx] = img

    def threshold(self, percentile_threshold=50, debug=False):
        """ Threshold out all pixels below the given percentile of the channel's intensity histogram. """
        self.X['thresh'] = dict()

        for chan_idx,img in self.X['bgsub'].items():
            nz = img > 0
            thresh = np.percentile(img[nz].ravel(), percentile_threshold)
            if debug:
                print(f"{self.df_channel.loc[chan_idx]['Label']} q{percentile_threshold}={thresh:0.3f}")
            q_img = img.copy() 
            q_img[q_img < thresh] = 0.
            self.X['thresh'][chan_idx] = q_img

    def denoise(self):
        self.X['denoise'] = dict()
        for chan_idx,img in self.X['thresh'].items():
            self.X['denoise'][chan_idx] = knn_denoise(img)

    def stats(self, transform='raw'):
        
        if transform == 'raw':
            chan_indices = self.included_channel_indices()
        else:
            chan_indices = self.X[transform].keys()

        df = {'channel_index':list(),
              'channel_label':list(),
              'sparsity':list()
              }
        for p in IMAGE_STATS_PERCENTILES:
            df[f'intensity_p{p}'] = list()
            df[f'grad_p{p}'] = list()
        
        for chan_idx in chan_indices:
            chan_info = self.df_channel.loc[chan_idx]
            istats = image_stats(self.X[transform][chan_idx],
                                 IMAGE_STATS_PERCENTILES)

            df['channel_index'].append(chan_idx)
            df['channel_label'].append(chan_info['Label'])
            df['sparsity'].append(istats['intensity_sparsity'])
            for k,p in enumerate(IMAGE_STATS_PERCENTILES):
                df[f'intensity_p{p}'].append(istats['intensity_percentiles'][k])
                df[f'grad_p{p}'].append(istats['grad_percentiles'][k])
            
        return pd.DataFrame(df)

    def preprocess(self, debug=False):
        self.bg_subtract(debug=debug)
        self.threshold(debug=debug)
        self.denoise()

    def write(self, output_dir, transform='denoise'):
        img_dir = os.path.join(output_dir, transform)
        if not os.path.exists(img_dir):
            os.makedirs(img_dir, exist_ok=True)
        images = self.X[transform]
        if isinstance(images, dict):
            chan_indices = images.keys()
        else:
            chan_indices = self.included_channel_indices()

        index2label = {k:lbl for lbl,k in self.label_to_index.items()}
        for chan_idx in chan_indices:
            chan_label = index2label[chan_idx]
            fname = os.path.join(img_dir, f"{chan_label}.tiff")
            img = self.X[transform][chan_idx].astype('float32')
            imsave(fname, img)
