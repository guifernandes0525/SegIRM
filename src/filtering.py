import numpy as np
import nibabel as nib
from skimage.segmentation import slic, find_boundaries, mark_boundaries
from scipy.ndimage import binary_dilation

def get_image_by_axis(img, axis = 'z'):
    x = img.shape[0] // 2
    y = img.shape[1] // 2
    z = img.shape[2] // 2

nii_path = "data_nii/"
fig_path = "segmentation/"

img = nib.load(nii_path + "img.nii").get_fdata()
gt = nib.load(nii_path + "gt.nii").get_fdata().astype(int)
mask = nib.load(nii_path + "mask.nii").get_fdata().astype(bool)

print("gt unique values", np.unique(gt))

print("img unique values", np.unique(img))

print("mask unique values", np.unique(mask))


# Horizontal slice

z = img.shape[2] // 2
img2d = img[:, :, z]
gt2d = gt[:, :, z]
mask2d = mask[:, :, z]

