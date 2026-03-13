import numpy as np
import skimage.exposure as se
from skimage.transform import rescale, resize
from skimage.filters import rank
from skimage.morphology import disk

class BioimgTools2d:

    def rescale(data: np.ndarray, scale_factor):
        return rescale(
        data,
        scale=scale_factor,  # e.g., 2.0 for 2x resolution
        order=0,  # Nearest-neighbor (no interpolation)
        mode='reflect',  # Edge handling
        preserve_range=True,  # Keep original intensity range
        anti_aliasing=False  # Disable for nearest-neighbor
    )

    def resize (data: np.ndarray, scale_d1, scale_d2):
            return resize(data, 
                   (data.shape[0]*scale_d1, data.shape[1]*scale_d2), 
                   order=0, 
                   mode='reflect', 
                   preserve_range=True,
                   anti_aliasing=False)

    def norm_histogram(data: np.ndarray):
        """Apply histogram equalization to each MRI slice."""
        normalized_data = np.zeros_like(data)
        normalized_data = se.equalize_hist(data)
        return normalized_data

    def adapt_norm_histogram(data):
        return se.equalize_adapthist(BioimgTools2d.normalize(data))

    def normalize(data: np.ndarray):
        """Normalize MRI to [0, max] range."""
        min = data.min()
        max = data.max()
        data -= min
        data /= max 
        return data

  
    #skimage.transform.resize_local_mean(image, output_shape, grid_mode=True, preserve_range=False, *, channel_axis=None)
    
    # Contrast stretching
    #p2, p98 = np.percentile(img, (2, 98))

    #img_rescale = rescale_intensity(img, in_range=(p2, p98))

    
    

