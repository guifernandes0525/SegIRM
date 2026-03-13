import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from Bioimages import Bioimage
from BioimgTools import BioimgTools2d
from BioSegmentation import BioSeg
from mri_metrics import (
    undersegmentation_error,
    boundary_recall,
    achievable_segmentation_accuracy,
)

def main():
    # Chemins des fichiers
    cwd = os.getcwd()
    nii_path = os.path.abspath(os.path.join(cwd, "data_nii"))
    fig_path = os.path.abspath(os.path.join(cwd, "segmentation"))
    metrics_path = os.path.abspath(os.path.join(cwd, "metrics", "2d_metrics.csv"))
    # Paramètres pour les tests
    n_segments_list = [100, 400, 1000]
    compactness_list = [0.1, 0.5, 1, 5]

    # Charger les données avec Bioimage
    mri = Bioimage(os.path.join(nii_path, "img.nii"))
    gt = Bioimage(os.path.join(nii_path, "gt.nii"))
    mask = Bioimage(os.path.join(nii_path, "mask.nii"))

    # Sélectionner une tranche 2D
    z = mri.shape[2] // 3
    mri_slice = BioimgTools2d.rescale(BioimgTools2d.norm_histogram(mri.get_slice(z, 'z'),10))
    gt_slice = BioimgTools2d.rescale(gt.get_slice(z, 'z'),10)
    mask_slice = BioimgTools2d.rescale(mask.get_slice(z, 'z'),10)

    # Ouvrir le fichier de métriques et écrire l'en-tête
    with open(metrics_path, 'w+') as metrics_file:
        metrics_file.write(
            "Method;N segments;Compactness;"
            "Under Segmentation;Boundary Recall;ASA\n"
        )

        # Boucle sur les combinaisons de paramètres
        for n_segments in n_segments_list:
            for compactness in compactness_list:
                # Segmentation SLIC avec BioSeg.simple_slic
                segments = BioSeg.simple_slic(
                    mri_slice,
                    n_superpix=n_segments,
                    compactness=compactness,
                    mask=mask_slice,
                )

                # Calcul des métriques
                ue = undersegmentation_error(segments, gt_slice, mask_slice)
                br = boundary_recall(segments=segments, gt=gt_slice, mask=mask_slice, radius=2)
                asa = achievable_segmentation_accuracy(segments, gt_slice, mask_slice)

                # Sauvegarder les métriques
                metrics_file.write(
                    f"BioSeg.simple_slic;{n_segments:4d};{compactness:.2f};"
                    f"{ue:.5f};{br:.5f};{asa:.5f}\n"
                )

                # Sauvegarder les frontières
                boundaries = mark_boundaries(image=mri_slice, label_img=segments, mode="inner")

                # Segmentation SLIC avec BioSeg.slic_zero_param
                segments_zero = BioSeg.slic_zero_param(mri_slice)

                # Calcul des métriques pour slic_zero_param
                ue_zero = undersegmentation_error(segments_zero, gt_slice, mask_slice)
                br_zero = boundary_recall(segments=segments_zero, gt=gt_slice, mask=mask_slice, radius=2)
                asa_zero = achievable_segmentation_accuracy(segments_zero, gt_slice, mask_slice)

                # Sauvegarder les métriques pour slic_zero_param
                metrics_file.write(
                    f"BioSeg.slic_zero_param;{n_segments:4d};{compactness:.2f};"
                    f"{ue_zero:.5f};{br_zero:.5f};{asa_zero:.5f}\n"
                )

                # Sauvegarder les frontières pour slic_zero_param
                boundaries_zero = mark_boundaries(image=mri_slice, label_img=segments_zero, mode="inner")

    # Visualisation d'un exemple de segmentation
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original MRI Slice")
    plt.imshow(mri_slice.T, cmap='gray')

    # Exemple de visualisation avec n_segments=400 et compactness=0.5
    segments_example = BioSeg.simple_slic(mri_slice, n_superpix=400, compactness=0.5, mask=mask_slice)
    boundaries_example = mark_boundaries(image=mri_slice, label_img=segments_example, mode="inner")[:,:,0]
    plt.subplot(1, 2, 2)
    plt.title("SLIC Segmentation Example")
    plt.imshow(boundaries_example.T, cmap='gray')

    plt.tight_layout()
    plt.show()
    plt.close()

def main_test_biotools():
    # Define paths
    img_path = "/home/guifernandes0525/Desktop/Enseirb/s8/EEL8-PROJ1/SegIRM/data_nii/"
    mri_path = 'img.nii'
    mask_path = 'mask.nii'

    # Load MRI and mask
    mri = Bioimage(img_path + mri_path)
    mask = Bioimage(img_path + mask_path)

    # Get a slice for testing
    slice_pos = mri.z_dim // 2
    mri_slice = mri.get_slice(slice_pos, 'z')
    mask_slice = mask.get_slice(slice_pos, 'z')

    # Visualize original MRI slice
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 4, 1)
    plt.title("Original MRI Slice")
    plt.imshow(mri_slice.T, cmap='gray')

    # Test rescale
    rescaled_slice = BioimgTools2d.rescale(mri_slice, scale_factor=2.0)
    plt.subplot(2, 4, 2)
    plt.title("Rescaled MRI Slice")
    plt.imshow(rescaled_slice.T, cmap='gray')

    # Test resize
    resized_slice = BioimgTools2d.resize(mri_slice, scale_d1=1.5, scale_d2=1.5)
    plt.subplot(2, 4, 3)
    plt.title("Resized MRI Slice")
    plt.imshow(resized_slice.T, cmap='gray')

    # Test histogram normalization
    norm_hist_slice = BioimgTools2d.norm_histogram(mri_slice)
    plt.subplot(2, 4, 4)
    plt.title("Histogram Normalized MRI Slice")
    plt.imshow(norm_hist_slice.T, cmap='gray')

    # Test adaptive histogram normalization
    adapt_norm_hist_slice = BioimgTools2d.adapt_norm_histogram(mri_slice)
    plt.subplot(2, 4, 5)
    plt.title("Adaptive Histogram Normalized MRI Slice")
    plt.imshow(adapt_norm_hist_slice.T, cmap='gray')

    # Test linear normalization
    linear_norm_slice = BioimgTools2d.normalize(mri_slice)
    plt.subplot(2, 4, 6)
    plt.title("Linear Normalized MRI Slice")
    plt.imshow(linear_norm_slice.T, cmap='gray')
    
    # Show all plots
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
