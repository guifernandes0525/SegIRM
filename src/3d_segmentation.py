import numpy as np
import nibabel as nib
from skimage.segmentation import slic, find_boundaries
from scipy.ndimage import binary_dilation

def undersegmentation_error(segments, gt, mask=None):
    if mask is None:
        mask = np.ones_like(gt, dtype=bool)

    total_error = 0
    total_pixels = mask.sum()

    for s in np.unique(segments):
        sp = (segments == s) & mask
        if sp.sum() == 0:
            continue

        _, counts = np.unique(gt[sp], return_counts=True)
        total_error += sp.sum() - counts.max()

    return total_error / total_pixels


def boundary_recall(segments, gt, mask=None, radius=2):
    gt_b = find_boundaries(gt, mode="thick")
    sp_b = find_boundaries(segments, mode="thick")

    if mask is not None:
        gt_b &= mask
        sp_b &= mask

    sp_b_dilated = binary_dilation(sp_b, iterations=radius)
    matched = gt_b & sp_b_dilated

    return matched.sum() / (gt_b.sum() + 1e-8)


nii_path = "data_nii/"
fig_path = "segmentation/"

img = nib.load(nii_path + "img.nii").get_fdata()
gt = nib.load(nii_path + "gt.nii").get_fdata().astype(int)
mask = nib.load(nii_path + "mask.nii").get_fdata().astype(bool)

n_segments_list = [1000, 3000, 6000]
compactness_list = [1, 10]
sigma_list = [0, 1]

with open(fig_path + '3d_metrics.csv', 'w+') as metrics:
    metrics.write('N segments;Compactness;σ (Gaussian Kernel diam.);Under Segmentation;Boundary Recall\n')


    for n_segments in n_segments_list:
        for compactness in compactness_list:
            for sigma in sigma_list:

                segments = slic(
                    img,
                    n_segments=n_segments,
                    compactness=compactness,
                    sigma=sigma,
                    mask=mask,
                    start_label=0,
                    channel_axis=None
                )

                ue = undersegmentation_error(segments, gt, mask)
                br = boundary_recall(segments, gt, mask, radius=2)

                print(
                    f"3D | n={n_segments:5d} | c={compactness:2d} | σ={sigma} "
                    f"| UE={ue:.4f} | BR={br:.4f}"
                )

                metrics.write(f"{n_segments:4d};{compactness:2d};{sigma};{ue:.5f};{br:.5f}\n")


                np.save(
                    f"{fig_path}/segments_3d_n{n_segments}_c{compactness}_s{sigma}.npy",
                    segments
                )
