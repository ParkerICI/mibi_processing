import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class PatchMaker(object):
    
    def __init__(self, image_size=(1024, 1024), kernel_size=256, stride=64):
        self.image_size = image_size
        self.unfold_params = {"kernel_size":kernel_size, "stride":stride}
        self.fold_params =  {"kernel_size":kernel_size, "stride":stride, "output_size":image_size}
        
        self.unfold_layer = nn.Unfold(**self.unfold_params)
        self.fold_layer = nn.Fold(**self.fold_params)
        self.correction_image = self.build_correction_image()
        

    def build_correction_image(self):
        # build a correction tensor that corrects for overlapping patches
        # when folding from an unfolded tensor
        ones_img = torch.ones([1, 1, self.image_size[0], self.image_size[1]])
        ones_img_flat = self.unfold_layer(ones_img)
        correction_image = self.fold_layer(ones_img_flat)
        return correction_image.reshape([1, 1, self.image_size[0], self.image_size[1]])
        
    def patch(self, X):
        """ Create a patchified tensor from X. X must be shape (nbatch, nchannels, height, width). """
        assert len(X.shape) == 4
        X_patch_flat = self.unfold_layer(X)
        X_patch = X_patch_flat.reshape([1, X.shape[1],
                                           self.unfold_params['kernel_size'],
                                           self.unfold_params['kernel_size'],
                                           -1])
        return X_patch
    
    def unpatch(self, X):
        """ Unpatch a tensor X. X must be shape (nbatch, nchannels, height, width, npatches). """
        assert len(X.shape) == 5
        nbatch, nchans, h, w, npatch = X.shape
        X_patch_flat = X.reshape([nbatch, nchans*h*w, npatch])
        X_unpatch = self.fold_layer(X_patch_flat)
        X_unpatch /= self.correction_image
        
        return X_unpatch
    
