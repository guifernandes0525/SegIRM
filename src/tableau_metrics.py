import numpy as np
import nibabel as nib
from skimage.segmentation import slic, watershed, felzenszwalb, mark_boundaries
from scipy.ndimage import sobel
from mri_metrics import (
    undersegmentation_error,
    boundary_recall,
    achievable_segmentation_accuracy
)
from skimage.measure import regionprops
from scipy.spatial import ConvexHull
from skimage.segmentation import find_boundaries



def shape_regulariy_SRC(segments, mask=None):

    labels = np.unique(segments)

    if mask is not None:
        total_pixels = mask.sum()
    else:
        total_pixels = segments.size

    SRC = 0

    for l in labels:

        S = (segments == l)

        if mask is not None:
            S = S & mask

        coords = np.column_stack(np.nonzero(S))

        if coords.shape[0] < 5:
            continue
        
        # -----------------------
        # |S|
        # -----------------------

        area = coords.shape[0]

        try:
            hull = ConvexHull(coords)

            hull_area = hull.volume
            hull_perimeter = hull.area

            SO = min(1, area / hull_area)

            boundary = find_boundaries(S, mode="inner")
            P = boundary.sum()

            CO = min(1, hull_perimeter / P) if P > 0 else 0

        except:
            SO = 0
            CO = 0

        # -----------------------
        # Vxy
        # -----------------------

        sigma_x = np.std(coords[:,0])
        sigma_y = np.std(coords[:,1])

        if max(sigma_x, sigma_y) > 0:
            Vxy = min(sigma_x, sigma_y) / max(sigma_x, sigma_y)
        else:
            Vxy = 0

        # -----------------------
        # SRC accumulation
        # -----------------------

        SRC += (area / total_pixels) * SO * Vxy * CO

    return SRC

nii_path = "C:/ENSEIRB/S8 PROJET_THEMATIQUE IRM/SegIRM/data_nii/"
fig_path = "C:/ENSEIRB/S8 PROJET_THEMATIQUE IRM/SegIRM/segmentation/"
metrics_path = "C:/ENSEIRB/S8 PROJET_THEMATIQUE IRM/SegIRM/metrics/"

# -----------------------------
# Load data
# -----------------------------
img = nib.load(nii_path + "img.nii").get_fdata()
gt = nib.load(nii_path + "gt.nii").get_fdata().astype(int)
mask = nib.load(nii_path + "mask.nii").get_fdata().astype(bool)

z = img.shape[2] // 3

img2d = img[:, :, z]
gt2d = gt[:, :, z]
mask2d = mask[:, :, z]

# -----------------------------
# Normalisation
# -----------------------------
img2d_norm = img2d.copy()

brain_pixels = img2d_norm[mask2d]
min_val = brain_pixels.min()
max_val = brain_pixels.max()

img2d_norm[mask2d] = (brain_pixels - min_val) / (max_val - min_val)

# -----------------------------
# Gradient feature
# -----------------------------
gx = sobel(img2d_norm, axis=0)
gy = sobel(img2d_norm, axis=1)
grad = np.sqrt(gx**2 + gy**2)

img_features = np.stack([img2d_norm, grad], axis=-1)

# -------------------------------------------------
# Parameter grids
# -------------------------------------------------

# SLIC
n_segments_list = [50, 100, 200, 300, 500, 800]
compactness_list = [5]
sigma_list = [1]

# Watershed
markers_list = [50, 100, 200, 300, 500]

# Felzenszwalb
scale_list = [50, 100, 200, 300]
fz_sigma_list = [0.5, 1]
min_size_list = [20, 50]

# -----------------------------
# Gradient for watershed
# -----------------------------
gradient = np.sqrt(gx**2 + gy**2)

# -----------------------------
# Metrics file
# -----------------------------
with open(metrics_path + "2d_metrics_all_methods.csv", "w+") as metrics:

    metrics.write(
        "Method;Param1;Param2;Param3;"
        "UnderSegmentation;BoundaryRecall;ASA;SRC\n"
    )

    # =========================================================
    # SLIC
    # =========================================================

    for n_segments in n_segments_list:
        for compactness in compactness_list:
            for sigma in sigma_list:

                segments = slic(
                    img_features,
                    n_segments=n_segments,
                    compactness=compactness,
                    sigma=sigma,
                    mask=mask2d,
                    start_label=0,
                    channel_axis=-1
                )

                ue = undersegmentation_error(segments, gt2d, mask2d)
                br = boundary_recall(segments, gt2d, radius=2)
                asa = achievable_segmentation_accuracy(segments, gt2d, mask2d)
                src = shape_regulariy_SRC(segments, mask2d)

                metrics.write(
                    f"SLIC;{n_segments};{compactness};{sigma};"
                    f"{ue:.5f};{br:.5f};{asa:.5f};{src:.5f}\n"
                )   

                boundaries = mark_boundaries(img2d_norm, segments, mode="inner")

                np.save(
                    f"{fig_path}slic_n{n_segments}_c{compactness}_s{sigma}.npy",
                    boundaries
                )

    # =========================================================
    # WATERSHED
    # =========================================================

    for markers in markers_list:

        segments = watershed(
            gradient,
            markers=markers,
            mask=mask2d
        )

        ue = undersegmentation_error(segments, gt2d, mask2d)
        br = boundary_recall(segments, gt2d, radius=2)
        asa = achievable_segmentation_accuracy(segments, gt2d, mask2d)
        src = shape_regulariy_SRC(segments, mask2d)

        metrics.write(
            f"Watershed;{markers};0;0;"
            f"{ue:.5f};{br:.5f};{asa:.5f};{src:.5f}\n"
        )

        boundaries = mark_boundaries(img2d_norm, segments, mode="inner")

        np.save(
            f"{fig_path}watershed_m{markers}.npy",
            boundaries
        )

    # =========================================================
    # FELZENSZWALB
    # =========================================================

    for scale in scale_list:
        for sigma in fz_sigma_list:
            for min_size in min_size_list:

                segments = felzenszwalb(
                    img2d_norm,
                    scale=scale,
                    sigma=sigma,
                    min_size=min_size
                )

                ue = undersegmentation_error(segments, gt2d, mask2d)
                br = boundary_recall(segments, gt2d, radius=2)
                asa = achievable_segmentation_accuracy(segments, gt2d, mask2d)
                src = shape_regulariy_SRC(segments, mask2d)

                metrics.write(
                    f"Felzenszwalb;{scale};{sigma};{min_size};"
                    f"{ue:.5f};{br:.5f};{asa:.5f};{src:.5f}\n"
                )

                boundaries = mark_boundaries(img2d_norm, segments, mode="inner")

                np.save(
                    f"{fig_path}fz_scale{scale}_sigma{sigma}_min{min_size}.npy",
                    boundaries
                )