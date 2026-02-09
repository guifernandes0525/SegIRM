import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import dilation

# solution to plot problem "sudo apt install python3-tk"

nii_path = "data_nii/"

nii_image = 'gt'

imgs_path = 'images/'

#image = nib.load(nii_path + nii_image + '.nii').get_fdata()

image = np.load("segmentation/boundaries_2d_n400_c1_s0.npy")

image = image[:,:,2]

image2 = np.load("segmentation/boundaries_2d_n100_c1_s0.npy")

print(image2.shape)
print(np.unique(image2))

image2 = image2[:,:,0]

dimensions = image.shape
#
#midx_slice = image[dimensions[0]//2,:,:]
#midy_slice = image[:,dimensions[1]//2,:]
#midz_slice = image[:,:,dimensions[2]//2]
#
#plane_slices = [midx_slice, midy_slice, midz_slice]

if nii_image == 'img' or nii_image == 'mask':
    map_color = 'gray'
else:
    map_color = 'jet'


plt.figure()
plt.imshow(image.T, cmap=map_color, origin = 'lower')
plt.title('Saggital slices of ' + nii_image)
plt.xlabel('x Axis')
plt.ylabel('y axis')
plt.colorbar(label='Signal intensity')
plt.show(block=False)

plt.figure()
plt.imshow(image2.T, cmap=map_color, origin = 'lower')
plt.title('Saggital slices of ' + nii_image)
plt.xlabel('x Axis')
plt.ylabel('y axis')
plt.colorbar(label='Signal intensity')
plt.show()

"""
plt.imshow(midx_slice.T, cmap=map_color, origin = 'lower')
plt.figure()
plt.title('Saggital slices of ' + nii_image)
plt.xlabel('y Axis')
plt.ylabel('z axis')
plt.colorbar(label='Signal intensity')
plt.show(block=False)
plt.savefig()


plt.figure()
plt.imshow(midy_slice.T, cmap=map_color, origin = 'lower')
plt.title('Coronal slices of '+ nii_image)
plt.xlabel('x Axis')
plt.ylabel('z axis')
plt.colorbar(label='Signal intensity')
plt.show(block=False)


plt.figure()
plt.imshow(midz_slice.T, cmap=map_color, origin = 'lower')
plt.title('Horizontal slices of '+ nii_image)
plt.xlabel('x Axis')
plt.ylabel('y axis')
plt.colorbar(label='Signal intensity')
plt.show()

 """