import numpy as np
import nibabel as nii
import scipy.io as scio

#Loading .nii
data_path = 'data/'
img = np.array(nii.load(data_path + 'img.nii').get_fdata()).astype(np.float32)
mask = np.array(nii.load(data_path + 'mask.nii').get_fdata()).astype(bool)
gt = np.array(nii.load(data_path + 'gt.nii').get_fdata()).astype(np.int)

#Loading from .mat
#img = scio.loadmat(data_path + 'img.mat')
#img = np.array(img['img']).astype(np.float32)
#gt = scio.loadmat(data_path + 'gt.mat')
#gt = np.array(gt['gt']).astype(np.int)
#mask = scio.loadmat(data_path + 'mask.mat')
#mask = np.array(mask['mask']).astype(bool)

#save in .npy
np.save(data_path + 'img.npy', img)
np.save(data_path + 'mask.npy', mask)
np.save(data_path + 'gt.npy', gt)

#load .npy
img = np.load(data_path + 'img.npy')
gt = np.load(data_path + 'gt.npy')
mask = np.load(data_path + 'mask.npy')
