import numpy as np
import nibabel as nib
from PIL import Image
import matplotlib.pyplot as plt


# Load the MRI image (replace with your file path)
file_path = "/home/guifernandes0525/Desktop/Enseirb/s8/EEL8-PROJ1/SegIRM/data_nii/img.nii"  # or .npy

img = nib.load(file_path)

print('shape of image is: ', img.shape, end='\n')

data = img.get_fdata()

print(img.header)

print('data type of the image', data.dtype)

data_mean = np.mean(data)
data_std = np.std(data)

print(f"The mean intensity of this image is {data_mean}, and the standard deviation is {data_std}.")

middle_slice = data[:, :, img.shape[-1] // 2 - 1]

plt.figure()
plt.imshow(middle_slice.T)
plt.show(block=False)

plt.figure()
plt.hist(np.ravel(data), bins=100)
plt.show()