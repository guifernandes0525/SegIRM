import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# solution to plot problem "sudo apt install python3-tk"

def plt_slice (slice) :

    plt.figure()
    plt.imshow(slice.T, cmap='gray', origin='lower')
    plt.xlabel('First axis')
    plt.ylabel('Second axis')
    plt.colorbar(label='Signal intensity')
    plt.show(block=False)

nii_path = "data_nii/"

mri_data = nib.load(nii_path + 'img.nii')

print(type(mri_data))
print(mri_data.shape)

header = mri_data.header
print(header.get_xyzt_units())

mri_image = mri_data.get_fdata()

dimensions = mri_image.shape

midx_slice = mri_image[dimensions[0]//2,:,:]
midy_slice = mri_image[:,dimensions[1]//2,:]
midz_slice = mri_image[:,:,dimensions[2]//2]

plane_slices = [midx_slice, midy_slice, midz_slice]

fig, (coronal, saggital, horizontal) = plt.subplots(1, 3)

# coronal -> front to back

coronal_img = coronal.imshow(midx_slice.T, cmap='gray', origin = 'lower')
coronal.set_title('Coronal slices')
coronal.set_xlabel('First Axis')
coronal.set_ylabel('Second axis')
fig.colorbar(mappable=coronal_img,label='Signal intensity', ax=coronal)

# saggital -> side to side

saggital_img = saggital.imshow(midy_slice.T, cmap='gray', origin = 'lower')
saggital.set_title('Saggital slices')
saggital.set_xlabel('First Axis')
saggital.set_ylabel('Second axis')
fig.colorbar(mappable=coronal_img,label='Signal intensity', ax=saggital)

# horizontal -> top to bottom

horizontal_img = horizontal.imshow(midz_slice.T, cmap='gray', origin = 'lower')
horizontal.set_title('Horizontal slice')
horizontal.set_xlabel('First Axis')
horizontal.set_ylabel('second Axis')
fig.colorbar(mappable = horizontal_img, label='Signal intensity', ax=horizontal)

plt.show()

print(plt.colormaps)
