import numpy as np
import nibabel as nib
from skimage.segmentation import slic, mark_boundaries
from mri_metrics import undersegmentation_error, boundary_recall, achievable_segmentation_accuracy


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

n_segments_list = [100, 400, 1000]
compactness_list = [1, 10, 20]
sigma_list = [0, 2]

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
                    img2d,
                    n_segments=n_segments,
                    compactness=compactness,
                    sigma=sigma,
                    mask=mask2d,
                    start_label=0,
                    channel_axis=None
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
                    f"{n_segments:4d};{compactness:2d};{sigma};"
                    f"{ue:.5f};{br:.5f};{asa:.5f}\n"
                )

                # --- Save boundaries ---
                boundaries = mark_boundaries(
                    image=img2d,
                    label_img=segments,
                    mode="inner"
                )

                np.save(
                    f"{fig_path}boundaries_2d_n{n_segments}_c{compactness}_s{sigma}.npy",
                    boundaries
                )
