import numpy as np
import nibabel as nib
from skimage.segmentation import slic, mark_boundaries
from scipy.ndimage import sobel
from mri_metrics import (
    undersegmentation_error,
    boundary_recall,
    achievable_segmentation_accuracy
)

nii_path = "data_nii/"
fig_path = "segmentation/"
metrics_path = "metrics/"


img = nib.load(nii_path + "img.nii").get_fdata()
gt = nib.load(nii_path + "gt.nii").get_fdata().astype(int)
mask = nib.load(nii_path + "mask.nii").get_fdata().astype(bool)


z = img.shape[2] // 3
img2d = img[:, :, z]
gt2d = gt[:, :, z]
mask2d = mask[:, :, z]

img2d_norm = img2d.copy()
brain_pixels = img2d_norm[mask2d]

min_val = brain_pixels.min()
max_val = brain_pixels.max()

img2d_norm[mask2d] = (brain_pixels - min_val) / (max_val - min_val)

# -------------------------------------------------
# 2) Add gradient magnitude as extra feature
# -------------------------------------------------
gx = sobel(img2d_norm, axis=0)
gy = sobel(img2d_norm, axis=1)
grad = np.sqrt(gx**2 + gy**2)

# Stack intensity + gradient → 2-channel feature image
img_features = np.stack([img2d_norm, grad], axis=-1)

# --- Parameter grid ---
n_segments_list = [100, 400, 1000]
compactness_list = [0.1, 0.5, 1, 5]   # Lower values → better boundary adherence
sigma_list = [0, 1, 2]

# --- Open metrics file ---
with open(metrics_path + "2d_metrics.csv", "w+") as metrics:

    metrics.write(
        "N segments;Compactness;Sigma;"
        "Under Segmentation;Boundary Recall;ASA\n"
    )

    for n_segments in n_segments_list:
        for compactness in compactness_list:
            for sigma in sigma_list:

                # --- SLIC segmentation ---
                segments = slic(
                    img_features,
                    n_segments=n_segments,
                    compactness=compactness,
                    sigma=sigma,
                    mask=mask2d,
                    start_label=0,
                    channel_axis=-1
                )

                # --- Metrics ---
                ue = undersegmentation_error(segments, gt2d, mask2d)
                br = boundary_recall(
                    segments=segments,
                    gt=gt2d,
                    mask=None,
                    radius=2
                )
                asa = achievable_segmentation_accuracy(
                    segments,
                    gt2d,
                    mask2d
                )

                # --- Save metrics ---
                metrics.write(
                    f"{n_segments:4d};{compactness:.2f};{sigma};"
                    f"{ue:.5f};{br:.5f};{asa:.5f}\n"
                )

                # --- Save boundaries ---
                boundaries = mark_boundaries(
                    image=img2d_norm,
                    label_img=segments,
                    mode="inner"
                )

                np.save(
                    f"{fig_path}boundaries_2d_n{n_segments}_c{compactness}_s{sigma}.npy",
                    boundaries
                )
