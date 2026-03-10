import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib

from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, watershed, slic
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
    
    #Extraction des contours GT et segmentation
    gt_b = find_boundaries(gt, mode="thick")
    sp_b = find_boundaries(segments, mode="thick")

    if mask is not None:
        gt_b &= mask
        sp_b &= mask

    #Dilatation
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
mask = mask[:, :, slice_id] > 0

print(np.sum(mask)/(mask.shape[0]*mask.shape[1]))

#plt.figure()
#plt.imshow(gt)
#plt.show()

#Normalisation de l'image
img = img / np.max(img)


#Réalisation des segmentations

segments_fz = felzenszwalb(img, scale = 50, sigma=0.5, min_size=50)
segments_slic = slic(img, n_segments=200, compactness=0.05, sigma=1, start_label=0, channel_axis = None)
gradient = sobel(img)
segments_watershed = watershed(gradient, markers= 50, compactness=0.01)

#Calcul des Metrics
br_fz = boundary_recall(segments_fz, gt)
ue_fz = undersegmentation_error(segments_fz, gt)

br_slic = boundary_recall(segments_slic, gt)
ue_slic = undersegmentation_error(segments_slic, gt)

br_ws = boundary_recall(segments_watershed, gt)
ue_ws = undersegmentation_error(segments_watershed, gt)

#Affichage des Metrics

print("\nFelzenszwalb")
print("Boundary Recall méthode Felzenszwalb :", br_fz)
print("Undersegmentation Error Felzenszwalb :", ue_fz)

print("\nWatershed")
print("Boundary Recall Watershed :", br_ws)
print("Undersegmentation Error Watershed:", ue_ws)

print("\nSLIC")
print("Boundary Recall SLIC :", br_slic)
print("Undersegmentation Error SLIC:", ue_slic)


print(f'SLIC number of segments: {len(np.unique(segments_slic))}')
print(f'Felzenszwalb number of segments: {len(np.unique(segments_fz))}')
print(f'Watershed number of segments: {len(np.unique(segments_watershed))}')

fig, ax = plt.subplots(2,2, figsize=(15, 5))

# Original image
ax[0,0].imshow(img, cmap="gray")
ax[0,0].set_title("Original IRM")

ax[0, 1].imshow(mark_boundaries(img, segments_slic))
ax[0, 1].set_title(
    f"SLIC\nBR={br_slic:.3f} UE={ue_slic:.3f}"
)

# Felzenszwalb
ax[1,0].imshow(mark_boundaries(img, segments_fz))
ax[1,0].set_title(
    f"Felzenszwalb\nBR={br_fz:.3f} UE={ue_fz:.3f}"
)

# Watershed
ax[1,1].imshow(mark_boundaries(img, segments_watershed))
ax[1,1].set_title(
    f"Watershed\nBR={br_ws:.3f} UE={ue_ws:.3f}"
)

for a in ax.ravel():
    a.set_axis_off()

plt.tight_layout()
plt.show()