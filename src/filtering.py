import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from skimage import exposure, filters
from skimage.util import img_as_ubyte

nii_path = "data_nii/"
fig_path = "segmentation/"

# Load data
img = nib.load(nii_path + "img.nii").get_fdata()
gt = nib.load(nii_path + "gt.nii").get_fdata().astype(int)
mask = nib.load(nii_path + "mask.nii").get_fdata().astype(bool)

# Print unique values for debugging
print("gt unique values:", np.unique(gt))
print("img unique values:", np.unique(img))
print("mask unique values:", np.unique(mask))

# Select a 2D slice (axial, coronal, or sagittal)
def get_2d_slices(mri, gt, mask, axis='z', slice_pos=None):
    if slice_pos is None:
        slice_pos = mri.shape[2] // 2  # Default to middle slice
    if axis == 'x':
        return mri[slice_pos, :, :], gt[slice_pos, :, :], mask[slice_pos, :, :]
    elif axis == 'y':
        return mri[:, slice_pos, :], gt[:, slice_pos, :], mask[:, slice_pos, :]
    else:  # 'z'
        return mri[:, :, slice_pos], gt[:, :, slice_pos], mask[:, :, slice_pos]

# Get 2D slices
img2d, gt2d, mask2d = get_2d_slices(img, gt, mask, axis='z')

# --- Preprocess for CLAHE ---
# Normalize to 0-1, then scale to 0-255 and convert to uint8
img2d_scaled = (img2d - img2d.min()) / (img2d.max() - img2d.min())
img2d_uint8 = img_as_ubyte(img2d_scaled)

# --- Contrast Enhancement Techniques ---

# 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
img2d_clahe = exposure.equalize_adapthist(img2d_uint8, clip_limit=0.03)

# 2. Windowing (MRI-specific intensity range)
window_min, window_max = np.percentile(img2d, 5), np.percentile(img2d, 95)
img2d_windowed = np.clip(img2d, window_min, window_max)
img2d_windowed = (img2d_windowed - window_min) / (window_max - window_min)

# 3. Log transform (avoid log(0) by adding 1)
img2d_log = np.log1p(img2d)
img2d_log = (img2d_log - img2d_log.min()) / (img2d_log.max() - img2d_log.min())

# 4. Sigmoid transform
def sigmoid(x, a=0.5, b=0.5):
    return 1 / (1 + np.exp(-a * (x - b)))
img2d_sigmoid = sigmoid(img2d)
img2d_sigmoid = (img2d_sigmoid - img2d_sigmoid.min()) / (img2d_sigmoid.max() - img2d_sigmoid.min())

# 5. Edge enhancement (Sobel)
img2d_edges = filters.sobel(img2d_windowed)
img2d_sharpened = img2d_windowed + 0.3 * img2d_edges
img2d_sharpened = np.clip(img2d_sharpened, 0, 1)

# --- Visualization ---
map_color = 'jet'
titles = [
    'Original MRI (scaled)',
    'CLAHE',
    'Windowed',
    'Log Transform',
    'Sigmoid Transform',
    'Sharpened (Sobel)'
]
images = [
    img2d_scaled,
    img2d_clahe,
    img2d_windowed,
    img2d_log,
    img2d_sigmoid,
    img2d_sharpened
]

plt.figure(figsize=(18, 12))
for i, (image, title) in enumerate(zip(images, titles), 1):
    plt.subplot(2, 3, i)
    if title == 'CLAHE':
        plt.imshow(image.T, cmap=map_color, origin='upper', vmin=0, vmax=255)
    else:
        plt.imshow(image.T, cmap=map_color, origin='upper', vmin=0, vmax=1)
    plt.title(title)
    plt.colorbar(label='Signal intensity')

# Plot ground truth and mask
plt.subplot(2, 3, 6)
plt.imshow(gt2d.T, cmap=map_color, origin='upper')
plt.title('Ground Truth')
plt.colorbar(label='Signal intensity')

plt.tight_layout()
plt.show()

# --- Histograms for comparison ---
plt.figure(figsize=(12, 8))
for i, (image, title) in enumerate(zip(images, titles), 1):
    plt.subplot(2, 3, i)
    if title == 'CLAHE':
        plt.hist(image.ravel(), bins=50, alpha=0.7, label=title, range=(0, 255))
    else:
        plt.hist(image.ravel(), bins=50, alpha=0.7, label=title, range=(0, 1))
    plt.title(title)
    plt.legend()
plt.show()
