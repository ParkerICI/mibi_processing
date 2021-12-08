import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as nn_init

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn_init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn_init.constant_(m.bias, 0.1)


class BGSubAndDenoiser(nn.Module):
    def __init__(self, num_preproc_channels=32):
        super(BGSubAndDenoiser, self).__init__()
        
        self.bg_preproc = nn.Conv2d(in_channels=1, out_channels=num_preproc_channels,
                                    kernel_size=3, padding=1)
        self.bg_preproc_act = nn.ReLU(inplace=True)
        
        self.chan_preproc = nn.Conv2d(in_channels=1, out_channels=num_preproc_channels,
                                      kernel_size=3, padding=1)
        self.chan_preproc_act = nn.ReLU(inplace=True)
        
        num_mix_channels = num_preproc_channels*2
        self.mix = nn.Conv2d(in_channels=2*num_preproc_channels, out_channels=num_mix_channels,
                             kernel_size=3, padding=1)
        self.mix_act = nn.ReLU(inplace=True)
        
        num_compression_channels = 1
        self.compress = nn.Conv2d(in_channels=num_mix_channels, out_channels=num_compression_channels,
                                     kernel_size=3, padding=0, stride=2)
        self.compress_act = nn.ReLU(inplace=True)
        
        self.out = nn.ConvTranspose2d(in_channels=num_compression_channels, out_channels=1,
                                      kernel_size=3, stride=2, padding=0, output_padding=1,
                                      dilation=1)

        self.out_act = nn.ReLU(inplace=True)

        self.apply(init_weights)

    def forward(self, chan_img, bg_img):
        #print('chan_img.shape=',chan_img.shape)
        #print('bg_img.shape=',bg_img.shape)
        
        chan_pp_img = self.chan_preproc_act(self.chan_preproc(chan_img))
        #print('chan_pp_img.shape=',chan_pp_img.shape)
        bg_pp_img = self.bg_preproc_act(self.bg_preproc(bg_img))
        #print('bg_pp_img.shape=',bg_pp_img.shape)
        
        combined_img = torch.cat([chan_pp_img, bg_pp_img], dim=1)
        
        #print('combined_img.shape=',combined_img.shape)
        
        mixed_img = self.mix_act(self.mix(combined_img))
        #print('mixed_img.shape=',mixed_img.shape)
        
        compressed_img = self.compress_act(self.compress(mixed_img))
        #print('compressed_img.shape=',compressed_img.shape)
        
        out_img = self.out_act(self.out(compressed_img))
        #print('out_img.shape=',out_img.shape)
        
        return out_img

    @staticmethod
    def load_from_file(file_path, device=None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = BGSubAndDenoiser()
        net.load_state_dict(torch.load(file_path, map_location=device))
        net.eval()
        return net

class BGSubtractRegression(nn.Module):

    def __init__(self, mask_matrix):
        super(BGSubtractRegression, self).__init__()

        self.num_channels = mask_matrix.shape[0]
        if torch.is_tensor(mask_matrix):
            self.M = mask_matrix
        else:
            self.M = torch.tensor(mask_matrix, dtype=torch.float32)

        W = np.random.rand(self.num_channels, self.num_channels)*mask_matrix.clone().detach().numpy()*1e-1
        W[0, 0] = 1. # gold channel should always be able to predict itself
        self.W = torch.nn.Parameter(torch.tensor(W.astype('float32')))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, img):
        """ Expects image of shape (nbatch, height, width, nchans) """
        return torch.matmul(img, self.relu(self.M*self.W))

    def get_comp_matrix(self):
        C = self.relu(self.M*self.W)
        return C.detach().numpy()

    @staticmethod
    def load_from_file(file_path, mask_matrix, device=None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = BGSubtractRegression(mask_matrix)
        net.load_state_dict(torch.load(file_path, map_location=device))
        net.eval()
        return net