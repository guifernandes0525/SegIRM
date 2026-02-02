import numpy as np
import nibabel as nii
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries

data_path = 'data/'
fig_path = 'figs/'

img = np.array(nii.load(data_path + 'img.nii').get_fdata()).astype(float)
mask = np.array(nii.load(data_path + 'mask.nii').get_fdata()).astype(bool)
gt = np.array(nii.load(data_path + 'gt.nii').get_fdata()).astype(int)

np.save(data_path + 'img.npy', img)
np.save(data_path + 'mask.npy', mask)
np.save(data_path + 'gt.npy', gt)

img = np.load(data_path + 'img.npy')
gt = np.load(data_path + 'gt.npy')
mask = np.load(data_path + 'mask.npy')

# Doing the 2D slice in the middle
print("Volume shape:", img.shape)

x = img.shape[0] // 2
y = img.shape[1] // 2
z = img.shape[2] // 2

# Sagittal (x fixed)
# img2d = img[x, :, :]
# gt2d  = gt[x, :, :]
# mask2d = mask[x, :, :]

# Coronal (y fixed)
# img2d = img[:, y, :]
# gt2d  = gt[:, y, :]
# mask2d = mask[:, y, :]

img2d = img[:, :, z]
gt2d  = gt[:, :, z]
mask2d = mask[:, :, z]

print("Slice shape:", img2d.shape)

#print("GT labels in mask:", np.unique(gt2d[mask2d]))
#print("Mask values:", np.unique(mask2d))

# Visualize image, ground truth and mask

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img2d, cmap='gray')
plt.title("Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(gt2d)
plt.title("Ground Truth")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(mask2d, cmap='gray')
plt.title("Mask")
plt.axis("off")

plt.tight_layout()
plt.savefig(fig_path + "2d_img_gt_mask.png", dpi=200)
plt.close()


number_of_segments = 200  # loop over values
compactness = 0.1         # sweep this too

segments = slic(
    img2d,
    n_segments=number_of_segments,
    compactness=compactness,
    mask=mask2d,
    start_label=1,
    channel_axis=None
).astype(np.int32)

plt.figure(figsize=(6, 6))
plt.imshow(mark_boundaries(img2d, segments))
plt.title(f"SLIC Superpixels (K={number_of_segments}, m={compactness})")
plt.axis("off")
plt.tight_layout()
plt.savefig(fig_path + "2d_superpixels_boundaries.png", dpi=200)
plt.close()

seg_m = segments[mask2d]
gt_m = gt2d[mask2d]

N = mask2d.sum()

asa_sum = 0
for s in np.unique(seg_m):
    gt_vals = gt_m[seg_m == s]
    if gt_vals.size == 0:
        continue
    _, counts = np.unique(gt_vals, return_counts=True)
    asa_sum += counts.max()

asa = asa_sum / N
print("ASA:", asa)

ue_sum = 0

# compute on masked pixels only (important!)
seg_m = segments[mask2d]
gt_m = gt2d[mask2d]
N = mask2d.sum()

for s in np.unique(seg_m):
    gt_vals = gt_m[seg_m == s]
    if gt_vals.size == 0:
        continue
    _, counts = np.unique(gt_vals, return_counts=True)
    max_overlap = counts.max()
    ue_sum += (gt_vals.size - max_overlap)  # |s| - max_g |s∩g|

ue = ue_sum / N
print("UE:", ue)
