import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib

from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, watershed
from skimage.segmentation import mark_boundaries
from scipy.ndimage import binary_dilation
from skimage.segmentation import find_boundaries 


#Pour les Metrics 

def undersegmentation_error(segments, gt, mask=None):
    if mask is None:
        mask = np.ones_like(gt, dtype=bool)

    total_error = 0
    total_pixels = mask.sum()

    for s in np.unique(segments): #Parcours de chaque segment
        sp = (segments == s) & mask
        if sp.sum() == 0:
            continue

        _, counts = np.unique(gt[sp], return_counts=True) #On compte combien de pixels du segment appartiennent à chaque label du GT
        total_error += sp.sum() - counts.max()

    return total_error / total_pixels #Divison par N

def boundary_recall(segments, gt, mask=None, radius=2):
    #Cette fonction calcule le BR égal au nombre de frontières GT détectées/Nombre de total de frontières GT
    
    gt_b = find_boundaries(gt, mode="thick")
    sp_b = find_boundaries(segments, mode="thick")

    if mask is not None:
        gt_b &= mask
        sp_b &= mask

    sp_b_dilated = binary_dilation(sp_b, iterations=radius)
    matched = gt_b & sp_b_dilated

    return matched.sum() / (gt_b.sum() + 1e-8)


#Chargement de l'image

nii_path = "../data_nii/"

img = nib.load(nii_path + "img.nii").get_fdata()
gt = nib.load(nii_path + "gt.nii").get_fdata() #Ground Truth
mask = nib.load(nii_path + "mask.nii").get_fdata().astype(bool)


print(img.shape)

#Découpe de l'image
slice_id = 90
img = img[:, :, slice_id]
gt = gt[:, :, slice_id]
mask = mask[:, :, slice_id]
#Normalisation de l'image
img = img / np.max(img)


#Réalisation des segmentations

segments_fz = felzenszwalb(img, scale=100, sigma=0.55, min_size=50)
#segments_slic = slic(img, n_segments=250, compactness=10, sigma=1, start_label=1, channel_axis=None)
#segments_quick = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)
gradient = sobel(img)
segments_watershed = watershed(gradient, markers=250, compactness=0.001, mask = mask)

#Calcul des Metrics
br_fz = boundary_recall(segments_fz, gt)
ue_fz = undersegmentation_error(segments_fz, gt)

br_ws = boundary_recall(segments_watershed, gt)
ue_ws = undersegmentation_error(segments_watershed, gt)

#Affichage des Metrics

print("\nFelzenszwalb")
print("Boundary Recall méthode Felzenszwalb :", br_fz)
print("Undersegmentation Error Felzenszwalb :", ue_fz)

print("\nWatershed")
print("Boundary Recall Watershed :", br_ws)
print("Undersegmentation Error Watershed:", ue_ws)



print(f'Felzenszwalb number of segments: {len(np.unique(segments_fz))}')
#print(f'SLIC number of segments: {len(np.unique(segments_slic))}')
#print(f'Quickshift number of segments: {len(np.unique(segments_quick))}')
print(f'Watershed number of segments: {len(np.unique(segments_watershed))}')

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# Original image
ax[0].imshow(img, cmap="gray")
ax[0].set_title("Original IRM")

# Felzenszwalb
ax[1].imshow(mark_boundaries(img, segments_fz))
ax[1].set_title(
    f"Felzenszwalb\nBR={br_fz:.3f} UE={ue_fz:.3f}"
)

# Watershed
ax[2].imshow(mark_boundaries(img, segments_watershed))
ax[2].set_title(
    f"Watershed\nBR={br_ws:.3f} UE={ue_ws:.3f}"
)

for a in ax.ravel():
    a.set_axis_off()

plt.tight_layout()
plt.show()