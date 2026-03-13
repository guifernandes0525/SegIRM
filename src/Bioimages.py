import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

class Bioimage:
    def __init__(self, path: str):
        self.path = path
        self.metadata = nib.load(self.path).header
        self.data = self._load_image()
        # add an iterator in order to go across slices
        self.shape = self.data.shape
        self.x_dim = self.data.shape[0]
        self.y_dim = self.data.shape[1]
        self.z_dim = self.data.shape[2]

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
        if len(self.data.shape) < 3:
            raise ValueError("No slicing into 2d ndarray") 

        if plane == 'x':
            return self.data[slice_pos, :, :]
        elif plane == 'y':
            return self.data[:, slice_pos, :]
        elif plane == 'z':
            return self.data[:, :, slice_pos]
        else:
            raise ValueError("Invalid plane")
        

# Example usage
if __name__ == "__main__":
    
    img_path = "/home/guifernandes0525/Desktop/Enseirb/s8/EEL8-PROJ1/SegIRM/data_nii/"
    mri_path = 'img.nii'
    mask_path = 'mask.nii'

    mri = Bioimage(img_path + mri_path)  
    mask = Bioimage(img_path + mask_path)

    plt.figure()
    
    plt.imshow(mri.get_slice(mri.z_dim//2, 'z').T)

    plt.show()
        
    plt.show()