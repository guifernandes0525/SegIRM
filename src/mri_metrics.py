import numpy as np

from skimage.segmentation import find_boundaries, mark_boundaries
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
    gt_b = find_boundaries(gt, mode="inner")
    sp_b = find_boundaries(segments, mode="inner")

    if mask is not None:
        ## should this be happening
        gt_b &= mask
        sp_b &= mask

    sp_b_dilated = binary_dilation(sp_b, iterations=radius)
    matched = gt_b & sp_b_dilated

    return matched.sum() / (gt_b.sum() + 1e-8)

def achievable_segmentation_accuracy(segments, gt, mask=None):
    
    """
    Compute Achievable Segmentation Accuracy (ASA)
    according to Achanta et al.

    Parameters
    ----------
    segments : ndarray (H, W)
        Superpixel label map.
    gt : ndarray (H, W)
        Ground-truth segmentation.
    mask : ndarray (H, W), optional
        Boolean mask of valid pixels.

    Returns
    -------
    asa : float
        Achievable Segmentation Accuracy.
    """

    if mask is not None:
        segments = segments[mask]
        gt = gt[mask]

    total_pixels = len(segments)
    asa_sum = 0

    # Iterate over each superpixel
    for sp_label in np.unique(segments):
        sp_mask = segments == sp_label
        gt_labels_in_sp = gt[sp_mask]

        if gt_labels_in_sp.size == 0:
            continue

        # Count overlap with each GT region
        counts = np.bincount(gt_labels_in_sp)
        asa_sum += counts.max()

    return asa_sum / total_pixels

