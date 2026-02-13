import numpy as np
import nibabel as nib
from skimage.segmentation import slic, find_boundaries, mark_boundaries
from scipy.ndimage import binary_dilation

import matplotlib.pyplot as plt

# TODO think about implementing things with cupy 

class MRI:
    # this is a constructor
    def __init__(self, path: str):
        self.image_path = path
        self._img_data = self._load()
        self._metadata = nib.load(self.image_path).get_fdata()
        self.plane 

    # TODO make sure that the types of the attributes are np
    def _load_image(self):
        if self.image_path.endswith('.nii'):
            self.img_data = nib.load(self.image_path).get_fdata()
        
        elif self.image_path.endswith('.npy'):
            self.img_data = np.load()
        
        else:
            raise(ValueError)
        
    def _load_metadata(self):
        if self.image_path.endswith('.nii'):
            self._metadata = nib.load(self.image_path).header
        else:
            raise(ValueError)
        
    def get_slice(self, slice_pos: int, plan: str = 'x'):
        if plan == 'x':
            return self._img_data[slice_pos,:,:]
        elif plan == 'y':
            return self._img_data[:,slice_pos,:]
        elif plan == 'z':
            return self._img_data[:,:,slice_pos]
        else:
            raise(ValueError)

    def get_volume(self):
        return self._image_data
        
class MRIVisualizer:
    def __init__(self, mri: MRI):
        # is this going to copy the whole structure? only reference to thing? 
        self.mri = mri

    def plot_slice(self, slice_pos: int, plane: str = 'x', cmap: str = 'gray'):
        """Plot a single slice."""
        slice_data = self.mri.get_slice(slice_pos, plane)
        plt.imshow(slice_data, cmap=cmap)
        plt.show()

    def plot_multiple_slices(self, slice_positions: list, plane: str = 'x'):
        """Plot multiple slices iteratively."""
        for pos in slice_positions:
            self.plot_slice(pos, plane)

    def savefig(img_path: str, save_path: str, *fig_paths):
        
        plt.figure()

        if not fig_paths:
            # only one figure to save


            image = np.load(fig_path)[:,:,0]
            plt.imshow(image, origin = 'lower')
            plt.title(fig_path.removesuffix('.npy').replace('_',' '))
            plt.xlabel('x Axis')
            plt.ylabel('y axis')
            plt.colorbar(label='Signal intensity')
            plt.tight_layout()
            plt.savefig(fig_path.removesuffix('.npy') + '.png')
            plt.clf()

    def plot_volume_by_cat(self):
        pass

class MRISegmentation:
    def __init__(self, mri: MRI):
        self.mri = mri
        self.segmentation_mask = None

    def apply_slic(self, n_segments: int = 100):
        """Apply SLIC segmentation."""
        volume = self.mri.get_volume()
        self.segmentation_mask = slic(volume, n_segments=n_segments)

    def get_mask(self):
        return self.segmentation_mask

