import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt


seg_imgs = "segmentation_imgs/" 
seg_npy = "segmentation/"

plt.figure()

for filename in os.listdir(seg_npy):
    file_path = os.path.join(seg_npy, filename)
    
    if os.path.isfile(file_path):

        image = np.load(file_path)[:,:,0]
        plt.imshow(image, origin = 'lower')
        plt.title(file_path.removesuffix('.npy').replace('_',' '))
        plt.xlabel('x Axis')
        plt.ylabel('y axis')
        plt.colorbar(label='Signal intensity')
        plt.tight_layout()
        plt.savefig(seg_imgs + filename.removesuffix('.npy') + '.png')
        plt.clf()

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