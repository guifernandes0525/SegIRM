import numpy as np
from skimage.segmentation import slic

#from mri_metrics import (undersegmentation_error, boundary_recall, achievable_segmentation_accuracy)


class BioSeg:
    
    # il faut voir le code source de slic et essayer de trouver des differentes implementations/ parallele + passive d'amelioration

    def simple_slic(data, n_superpix = 256, compactness = 1, mask = None, max_num_iter=10, enforce_connectivity = True):
        return slic(data, 
                    n_segments=n_superpix, 
                    compactness=compactness, 
                    max_num_iter=max_num_iter, 
                    sigma=0, 
                    spacing=None, 
                    convert2lab=False, 
                    enforce_connectivity=enforce_connectivity, 
                    slic_zero=False,  
                    mask=mask, 
                    channel_axis=None)

    def slic_zero_param(data):
        return slic(data, channel_axis=None, slic_zero=True)