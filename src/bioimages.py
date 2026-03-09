import numpy as np
import nibabel as nib
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# Image treatment
from skimage import (exposure, feature, filters, io, measure,
                      morphology, restoration, segmentation, transform,
                      util)

class Bioimage:
    def __init__(self, path: str):
        self.path = path
        self.data = self._load_image()
        self.metadata = nib.load(self.path).header
        self.dims = self.data.shape

    def _load_image(self):
        """Load MRI data using nibabel or numpy."""
        if self.path.endswith('.nii'):
            return (nib.load(self.path).get_fdata())
        elif self.path.endswith('.npy'):
            return np.load(self.path)
        else:
            raise ValueError("Unsupported file format")
    
    def describe(self):
        return {"dimensions": self.dims,
                "mean" : np.mean(self.data),
                "std deviation" : np.std(self.data),
                "min max": [np.min(self.data), np.max(self.data)]}
        
    def get_slice(self, slice_pos: int, plane: str = 'z'):
        # Slice directly using NumPy
        if plane == 'x':
            return self.data[slice_pos, :, :]
        elif plane == 'y':
            return self.data[:, slice_pos, :]
        elif plane == 'z':
            return self.data[:, :, slice_pos]
        else:
            raise ValueError("Invalid plane")
        

class Bioimg:
    def __init__(self, bioimg: Bioimage):
            self.bioimg = bioimg

    def show_slice_pil(self, slice_pos: int, plane: str = 'z', interval = range):
        """Fast display using PIL and NumPy."""
        # Slice directly using NumPy
        slice_data = self.bioimg.get_slice(slice_pos=slice_pos,plane='x')
        

        normalized = self._normalize_slice(slice_data)
        Image.fromarray(normalized, mode='L').show()

    def _normalize_slice(bioimage: Bioimage, cut_plan = 'z') -> np.ndarray:
        """Normalize slice data to 0-255 using NumPy."""
        slice_data = slice_data.astype(np.float64)
        slice_data -= self.data.min()
        slice_data /= self.data.max()
        return (self.data * 255)

    def show_slice_pil(data, slice_pos: int, plane: str = 'z'):
        """Fast display using PIL and NumPy."""
        # Slice directly using NumPy
        if plane == 'x':
            slice_data = self.data[slice_pos, :, :]
        elif plane == 'y':
            slice_data = self.data[:, slice_pos, :]
        elif plane == 'z':
            slice_data = self.data[:, :, slice_pos]
        else:
            raise ValueError("Invalid plane")

        normalized = self._normalize_slice(slice_data)
        Image.fromarray(normalized, mode='L').show()

    def save_slice_pil(self, slice_pos: int, plane: str = 'z', save_path: str = 'slice.png'):
        """Fast save using PIL and NumPy."""
        # Slice directly using NumPy
        if plane == 'x':
            slice_data = self.data[slice_pos, :, :]
        elif plane == 'y':
            slice_data = self.data[:, slice_pos, :]
        elif plane == 'z':
            slice_data = self.data[:, :, slice_pos]
        else:
            raise ValueError("Invalid plane")

        normalized = self._normalize_slice(slice_data)
        Image.fromarray(normalized, mode='L').save(save_path)
        print(f"Saved: {save_path}")

    def plot_slice_mpl(self, slice_pos: int, plane: str = 'z', cmap: str = 'gray'):
        """Display with matplotlib (for annotations)."""
        # Slice directly using NumPy
        if plane == 'x':
            slice_data = self.data[slice_pos, :, :]
        elif plane == 'y':
            slice_data = self.data[:, slice_pos, :]
        elif plane == 'z':
            slice_data = self.data[:, :, slice_pos]
        else:
            raise ValueError("Invalid plane")

        plt.imshow(slice_data, cmap=cmap)
        plt.title(f"MRI Slice (plane={plane}, pos={slice_pos})")
        plt.axis('off')
        plt.colorbar(label='Intensity')
        plt.show()

# Example usage
if __name__ == "__main__":
    mri = Bioimage("/home/guifernandes0525/Desktop/Enseirb/s8/EEL8-PROJ1/SegIRM/data_nii/img.nii")  # Replace with your file
    print(mri.get_stats())
    visualizer = Bioimgshow2d(mri.get_data)

    # Fast display with PIL (z-plane, slice 10)
    visualizer.show_slice_pil(10, plane='z')

    # Save with PIL (z-plane, slice 10)
    visualizer.save_slice_pil(10, plane='z', save_path='slice_pil.png')

    # Display with matplotlib (z-plane, slice 10)
    visualizer.plot_slice_mpl(10, plane='z')

    visualizer.plot_slice_mpl(10, plane='x')



 
#   load image, mask and gt
#   pre processing (filtering, resize)
#   segmentation & metrics obtention
#   plotting of graphs and results 
#   comparison with 3d slic 
#   