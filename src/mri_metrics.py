import numpy as np

from skimage.segmentation import find_boundaries, mark_boundaries
from scipy.ndimage import binary_dilation

def undersegmentation_error(segments, gt, mask=None):
    if mask is None:
        mask = np.ones_like(gt, dtype=bool)

    # Ensure segments and mask are boolean or integer
    segments = np.asarray(segments, dtype=int)
    mask = np.asarray(mask, dtype=bool)

    total_error = 0
    total_pixels = mask.sum()

    for s in np.unique(segments):
        sp = (segments == s) & mask  # Now segments and mask are compatible
        if sp.sum() == 0:
            continue

        _, counts = np.unique(gt[sp], return_counts=True)
        total_error += sp.sum() - counts.max()

    return total_error / total_pixels
from skimage.segmentation import find_boundaries
from scipy.ndimage import binary_dilation

def boundary_recall(segments, gt, mask=None, radius=2):
    # Ensure all arrays are of the correct type
    segments = np.asarray(segments, dtype=int)
    gt = np.asarray(gt, dtype=int)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)

    gt_b = find_boundaries(gt, mode="inner")
    sp_b = find_boundaries(segments, mode="inner")

    if mask is not None:
        gt_b &= mask
        sp_b &= mask

    sp_b_dilated = binary_dilation(sp_b, iterations=radius)
    matched = gt_b & sp_b_dilated

    return matched.sum() / (gt_b.sum() + 1e-8)


def achievable_segmentation_accuracy(segments, gt, mask=None):
    # Ensure all arrays are of the correct type
    segments = np.asarray(segments, dtype=int)
    gt = np.asarray(gt, dtype=int)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        segments = segments[mask]
        gt = gt[mask]

    total_pixels = len(segments)
    if total_pixels == 0:
        return 0.0

    asa_sum = 0

    for sp_label in np.unique(segments):
        sp_mask = segments == sp_label
        gt_labels_in_sp = gt[sp_mask]

        if gt_labels_in_sp.size == 0:
            continue

        counts = np.bincount(gt_labels_in_sp)
        if len(counts) > 0:
            asa_sum += counts.max()

    return asa_sum / total_pixels
