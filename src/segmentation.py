import numpy as np
import nibabel as nii
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries


def get_image_by_axis(img, axis = 'z'):
    x = img.shape[0] // 2
    y = img.shape[1] // 2
    z = img.shape[2] // 2

    if axis == 'x':
        return img.shape[0], img[x, :, :], gt[x, :, :], mask[x, :, :]
    if axis == 'y':
        return img.shape[1], img[:, y, :], gt[:, y, :], mask[:, y, :]
    if axis == 'z': 
        return img.shape[2], img[:, :, z], gt[:, :, z], mask[:, :, z]

nii_path = 'data_nii/'
npy_path = 'data_npy/'
fig_path = 'segmentation/'

img = np.array(nii.load(nii_path + 'img.nii').get_fdata()).astype(float)
mask = np.array(nii.load(nii_path + 'mask.nii').get_fdata()).astype(bool)
gt = np.array(nii.load(nii_path + 'gt.nii').get_fdata()).astype(int)

np.save(npy_path + 'img.npy', img)
np.save(npy_path + 'mask.npy', mask)
np.save(npy_path + 'gt.npy', gt)

img = np.load(npy_path + 'img.npy')
gt = np.load(npy_path + 'gt.npy')
mask = np.load(npy_path + 'mask.npy')

# Doing the 2D slice in the middle
print("Volume shape:", img.shape)

x = img.shape[0] // 2
y = img.shape[1] // 2
z = img.shape[2] // 2

shape, img2d, gt2d, mask2d = get_image_by_axis(img)

n_segments = 15*15  # loop over values
compactness = 10    # sweep this too

# we can play with the sigma (gaussian smoothing kernel) to filter the image
segments = slic(image=img2d, mask=mask2d, segments=n_segments)

# compute on masked pixels only (important!)
seg_m = segments[mask2d]
gt_m = gt2d[mask2d]
N = mask2d.sum()


    