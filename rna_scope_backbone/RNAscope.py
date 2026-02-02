import os
from typing import Tuple
from skimage.measure import label as cc_label
from scipy.ndimage import distance_transform_edt, map_coordinates
import blosc
import h5py
import tifffile
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy import ndimage
from scipy.optimize import linear_sum_assignment, curve_fit
from tqdm import tqdm
from scipy.interpolate import UnivariateSpline
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
import numpy as np

from skimage.measure import regionprops

from skimage import  filters
from skimage.registration import phase_cross_correlation
from scipy.ndimage import binary_dilation
from skimage.morphology import disk, binary_opening,binary_closing,remove_small_holes
from scipy.ndimage import binary_fill_holes, gaussian_filter, label as ndi_label

from rna_scope_backbone.predict_htt import function_htt
from rna_scope_backbone.spot_detection import SMLMSpotDetector
from rna_scope_backbone.torch_stuff_DL import Gaussian_flexsigma, LM_MLE_with_iter, Gaussian3DPSF, LM_MLE_for_zstack
import scienceplots
plt.style.use('science')
from scipy.ndimage import gaussian_filter, shift, center_of_mass
from skimage.transform import AffineTransform, matrix_transform
import imreg_dft as ird
from skimage import exposure, measure
import napari
from scipy.stats import chi2
import torch.nn.functional as F
import math
def show_napari_images_lables(raw_image, label_image,label_image2, title="Image and Labels"):
    """
    Displays raw and label images side by side in Napari.

    Parameters:
    - raw_image (np.ndarray): The raw MIP image.
    - label_image (np.ndarray): The corresponding label image.
    - title (str): Title of the Napari window.
    """
    with napari.gui_qt():
        viewer = napari.Viewer(title=title)
        viewer.add_image(raw_image, name="Raw Image", colormap='gray', blending='additive')
        viewer.add_labels(label_image, name="Labels", blending='additive')
        viewer.add_labels(label_image2, name="Labels", blending='additive')

def show_napari(image):
    import napari
    with napari.gui_qt():
        viewer = napari.view_image(image, title="Detected ROIs")

def show_napari_spots(image, spots):
    import napari
    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(image, name="Summed Images")
        points = np.array([[frame, y , x ] for frame, y, x in spots])
        viewer.add_points(points,  face_color='transparent', name="spots")

def show_tensor_spots(image, spots):
    import napari
    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(image.detach().cpu().numpy(), name="Summed Images")
        points = np.array([[ y, x] for y, x in spots])
        viewer.add_points(data=points, face_color='transparent', name="spots")

def show_tensor(image):
    import napari
    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(image.detach().cpu().numpy(), name="Summed Images")

class RNAscopeclass:
    def __init__(self, cfg):
        """
        Example initializer that uses a color-based channel map
        instead of separate fields like 'full_length_HTT_channel', etc.
        """
        self.bright_image_dir  = cfg['bright_image_dir']
        self.dark_image_path   = cfg['dark_image_path']
        self.generate_plots    = cfg['generate_plots']

        self.num_channels      = cfg['num_channels']
        self.channel_map       = cfg['channel_map']
        self.roisize           = cfg['roisize']
        self.iterations        = cfg['iterations']
        self.dev               = cfg['dev']
        self.config_paths      = cfg['config_paths']  # Dictionary of config paths per mode/key


    def process_labelsindapi_intensityelse(self,filtered_labels_dapi: np.ndarray,
                                 image_firstexon: np.ndarray) -> Tuple[np.ndarray, list]:
        """
        Process labeled regions and calculate surrounding intensities in 3D.

        Args:
            filtered_labels_dapi: 3D array of labeled regions (0 = background)
            image_firstexon: 3D image data for intensity calculations

        Returns:
            Tuple containing:
            - combined_surrounding_mask: 3D boolean mask of all analyzed regions
            - mean_intensity_around_dapi: List of mean intensities for each label
        """
        if filtered_labels_dapi is None:
            return None, None
        unique_labels = np.unique(filtered_labels_dapi)
        combined_surrounding_mask = np.zeros_like(filtered_labels_dapi, dtype=bool)
        mean_intensity_around_dapi = []

        # Convert image to photons once outside the loop
        photon_image = self.convert_to_photons(image_firstexon)

        for label_idx in unique_labels:
            if label_idx == 0:  # Skip background
                continue

            mask = filtered_labels_dapi == label_idx
            mask_size = np.sum(mask)

            if mask_size == 0:
                print(f'Skipped label {label_idx} - empty mask')
                mean_intensity_around_dapi.append(0)
                continue

            #print(f'\nProcessing label {label_idx}, mask size = {mask_size}')

            # Calculate surrounding intensity
            result = self.calculate_surrounding_median_3D(
                photon_image,
                mask,
                border_width=15,
                include_maskitself=True,
                use_mean=True
            )
            if result is None:
                combined_surrounding_mask, mean_intensity_around_dapi = None,None
            else:

                mean_surrounding, cropped_surrounding_mask, (min_z, max_z, min_row, max_row, min_col, max_col) = result
                # Update combined mask if valid coordinates
                if (max_z > min_z) and (max_row > min_row) and (max_col > min_col):
                    combined_surrounding_mask[min_z:max_z, min_row:max_row, min_col:max_col] |= cropped_surrounding_mask
                mean_intensity_around_dapi.append(mean_surrounding)
        return combined_surrounding_mask, mean_intensity_around_dapi
    def label3d_to_sparse_dict(self,label_array):
        """
        Converts a 3D label array into a sparse dictionary with uint16 storage.

        Args:
            label_array (np.ndarray): Input 3D array of shape (depth, height, width).

        Returns:
            dict: Sparse dictionary with keys:
                  - "coords": uint16 array of shape (N, 3) storing non-zero coordinates.
                  - "labels": uint16 array of shape (N,) storing non-zero labels.
                  - "shape": uint16 array storing original 3D shape.

        Raises:
            ValueError: If input is not 3D or dimensions exceed uint16 limits.
        """
        if len(label_array.shape) != 3:
            raise ValueError("Input array must be 3D.")

        # Check if dimensions fit within uint16 range (0–65535)
        original_shape = label_array.shape
        if any(dim > 65535 for dim in original_shape):
            raise ValueError("Array dimensions must be ≤65535 to use uint16 storage.")

        # Extract non-zero coordinates and labels
        non_zero_coords = np.argwhere(label_array != 0)
        non_zero_labels = label_array[non_zero_coords[:, 0], non_zero_coords[:, 1], non_zero_coords[:, 2]]

        # Ensure labels fit in uint16 (user claims ≤63500)
        if np.any(non_zero_labels > 65535):
            raise ValueError("Labels exceed uint16 maximum (65535).")

        # Convert to uint16
        return {
            "coords": non_zero_coords.astype(np.uint16),
            "labels": non_zero_labels.astype(np.uint16),
            "shape": np.array(original_shape, dtype=np.uint16)
        }

    def sparse_dict_to_label3d(self,sparse_dict):
        """
        Reconstructs the original 3D label array from a sparse dictionary.

        Args:
            sparse_dict (dict): Dictionary with keys "coords", "labels", "shape".

        Returns:
            np.ndarray: Reconstructed 3D label array of dtype uint16.
        """
        # Extract components
        coords = sparse_dict["coords"]
        labels = sparse_dict["labels"]
        original_shape = tuple(sparse_dict["shape"].astype(int))

        # Reconstruct the 3D array
        reconstructed = np.zeros(original_shape, dtype=np.uint16)
        reconstructed[coords[:, 0], coords[:, 1], coords[:, 2]] = labels

        return reconstructed


    # Function to filter labels by size
    def filter_labels_by_size(self, label_image, min_size=0, max_size=np.inf):
        label_sizes = np.bincount(label_image.ravel())

        if self.generate_plots==True:
            plt.hist(label_sizes,range=(0, 50000),bins=100)
            plt.xlabel('Label size')
            plt.ylabel('Probability')
            plt.show()
        filtered_image = np.zeros_like(label_image)
        for label_index, size in enumerate(label_sizes):
            if size > min_size and size< max_size:
                filtered_image[label_image == label_index] = label_index
        return filtered_image

    def compute_gain(self, images_max=np.inf):
        # If bright_image_dir and dark_image_path are numbers, use them as gain and offset
        if isinstance(self.bright_image_dir, (int, float)) and isinstance(self.dark_image_path, (int, float)):
            self.gain_offset = (self.bright_image_dir, self.dark_image_path)
            return  # Skip further processing, as the gain/offset are already provided

        # List all TIFF files in the bright image directory
        bright_image_files = [f for f in os.listdir(self.bright_image_dir) if f.endswith('.tif')]
        means = []
        variances = []

        # Read the dark image once
        dark_image = tifffile.imread(self.dark_image_path)
        varbg = np.var(dark_image)
        self.dark_mean = np.mean(dark_image)
        iterr = 0
        for bright_image_file in tqdm(bright_image_files, desc='Compute photon gain from multiple image series...'):
            if iterr >= images_max:
                break
            bright_image_path = os.path.join(self.bright_image_dir, bright_image_file)

            # Read the bright image
            bright_image = tifffile.imread(bright_image_path)

            bg_corrected = np.clip(bright_image - self.dark_mean, 0, np.inf)

            variance = np.var(bg_corrected, 0)
            mean = np.mean(bg_corrected, 0)

            means.append(mean.flatten())
            variances.append(variance.flatten())
            iterr += 1

        mean = np.concatenate(means)
        variance = np.concatenate(variances)

        # Quantile-based binning to ensure each bin has data points
        num_bins = 200
        quantiles = np.linspace(0, 1, num_bins + 1)
        bin_edges = np.quantile(mean, quantiles)
        bin_means = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        bin_indices = np.digitize(mean, bin_edges) - 1

        valid_bins = bin_indices[(bin_indices >= 0) & (bin_indices < num_bins)]
        unique_valid_bins = np.unique(valid_bins)
        bin_means = bin_means[unique_valid_bins]
        bin_variances = [np.mean(variance[bin_indices == i]) for i in unique_valid_bins]

        # Fit a spline
        self.photongain_spline = UnivariateSpline(bin_means, bin_variances, s=1e8, k=1)  # k=1 for linear spline
        self.x_hat = np.linspace(bin_means.min(), bin_means.max(), 10000)
        y_hat = self.photongain_spline(self.x_hat)

        # Calculate the slope (gain) of the spline
        slope_hat = self.photongain_spline.derivative()(self.x_hat)

        if self.generate_plots:
            # Plotting
            plt.figure(figsize=(3, 3), dpi=300)
            plt.scatter(bin_means, bin_variances, s=10, label='Binned Data', alpha=0.5)
            plt.plot(self.x_hat, y_hat, label='Spline Fit', color='red')
            plt.xlabel('Mean ADU [ADU]')
            plt.ylabel(r'Variance ADU [$\text{ADU}^2$]')
            plt.legend()
            plt.tight_layout(pad=0.4)
            plt.show()

            plt.figure(figsize=(3, 3), dpi=300)
            plt.plot(self.x_hat, 1 / slope_hat, marker='o', linestyle='-', label='Gain')
            plt.xlabel('Mean ADU')
            plt.ylabel('Gain [Photons/ADU]')
            plt.legend()
            plt.tight_layout(pad=3)
            plt.yscale('log')
            plt.show()

    def convert_to_photons(self, nd_array):
        if not hasattr(self, 'gain_offset') or self.gain_offset is None:
            if self.photongain_spline is None or self.x_hat is None:
                raise ValueError("compute_gain must be called before convert_to_photons.")

            # Subtract the dark mean from the input array and convert to float
            bg_corrected = np.clip(nd_array.astype(np.float64) - self.dark_mean, 0, np.inf)

            # Use the spline derivative to get the gain
            gain = self.photongain_spline.derivative()

            # Apply the gain to the ND array
            gain_values = gain(bg_corrected)

            # Handle potential division by zero or invalid gain values
            gain_values = np.where(gain_values <= 0, np.finfo(np.float64).eps, gain_values)

            converted_array = (1 / gain_values) * bg_corrected

        else:
            # Use pre-computed gain and offset
            bg_corrected = np.clip(nd_array.astype(np.float64) - self.gain_offset[1], 0, np.inf)
            converted_array = self.gain_offset[0] * bg_corrected

        return converted_array

    # def convert_to_photons(self, nd_array):
    #     if self.photongain_spline is None or self.x_hat is None:
    #         raise ValueError("compute_gain must be called before convert_to_photons.")
    #
    #     # Subtract the dark mean from the input array
    #     bg_corrected = np.clip(nd_array - self.dark_mean, 0, np.inf)
    #
    #     converted_array = self.photongain_spline(bg_corrected)
    #
    #     # Apply the function to the ND array
    #     #converted_array = bg_corrected*0.45
    #     return converted_array

    def load_image(self, file_path, channel_to_load=None):
        ext = os.path.splitext(file_path)[1]
        channel_name = None
        if ext == '.ims':
            with h5py.File(file_path, 'r') as f:
                dataset = f['DataSet']['ResolutionLevel 0']['TimePoint 0']
                image = []

                if channel_to_load is not None:
                    # Load only the specified channel
                    channel_data = dataset[f'Channel {channel_to_load}']['Data'][...]
                    valid_slices = [slice for slice in channel_data if np.any(slice != 0)]
                    image = np.array(valid_slices)
                else:
                    # Load all channels if channel_to_load is not specified
                    for i in range(self.num_channels):
                        channel_data = dataset[f'Channel {i}']['Data'][...]
                        valid_slices = [slice for slice in channel_data if np.any(slice != 0)]
                        image.append(np.array(valid_slices))
                    image = np.stack(image, axis=0)

        elif ext == '.npz':
            # Load compressed data and metadata from .npz file
            with np.load(file_path, allow_pickle=True) as data:
                compressed_data = data['compressed_data']
                metadata = data['metadata'].item()  # Extract metadata dictionary
            # Decompress the data
            decompressed_data = blosc.decompress(compressed_data)

            # Convert back to NumPy array
            shape = tuple(metadata['shape'])
            dtype = np.dtype(metadata['dtype'])
            image = np.frombuffer(decompressed_data, dtype=dtype).reshape(shape)

        elif ext == '.h5':
            with h5py.File(file_path, 'r') as f:
                image = f['image'][()]

        elif ext == '.tif' or ext == '.tiff':
            image = tifffile.imread(file_path)

        else:
            print(f"Unknown file extension: {ext}")
            return None

        # Handle channel_to_load for all file types
        if channel_to_load is not None:
            image = self._select_channel(image, channel_to_load)
            if metadata is not None:

                channel_name = metadata['channels'][channel_to_load]
            else:
                channel_name = None
        return image,channel_name

    def _select_channel(self, image, channel_to_load):
        """
        Selects the specified channel from the image data.

        Parameters:
        - image (numpy.ndarray): The image data.
        - channel_to_load (int): The channel index to load.

        Returns:
        - numpy.ndarray: The image data for the specified channel.
        """
        if image.ndim == 4:
            # Attempt to determine the channel axis
            # Common formats are (channels, z, y, x) or (z, y, x, channels)
            if image.shape[0] == self.num_channels:
                # Channels along axis 0
                image = image[channel_to_load]
            elif image.shape[-1] == self.num_channels:
                # Channels along last axis
                image = image[..., channel_to_load]
            else:
                raise ValueError("Cannot determine channel axis in image data.")
        elif image.ndim == 3:
            if self.num_channels == 1:
                # Single-channel data, nothing to select
                pass
            else:
                raise ValueError("Data has 3 dimensions but num_channels > 1. Cannot select channel.")
        else:
            raise ValueError("Unexpected image dimensions.")
        return image

    def calculate_surrounding_median_3D(self, image_HTT, mask, border_width=20,include_maskitself=False, use_mean=False):
        mask_indices = np.argwhere(mask)
        if mask_indices.size == 0:
            print('Empty mask, skipping...')
            return None  # Avoid processing empty masks

        min_z, min_row, min_col = np.min(mask_indices, axis=0)
        max_z, max_row, max_col = np.max(mask_indices, axis=0)

        # Expand the bounding box by border_width, ensuring it stays within image bounds
        min_z = max(0, min_z - border_width)
        max_z = min(image_HTT.shape[0], max_z + border_width + 1)
        min_row = max(0, min_row - border_width)
        max_row = min(image_HTT.shape[1], max_row + border_width + 1)
        min_col = max(0, min_col - border_width)
        max_col = min(image_HTT.shape[2], max_col + border_width + 1)

        # Crop the mask and image to the expanded bounding box
        cropped_mask = mask[min_z:max_z, min_row:max_row, min_col:max_col]
        cropped_image = image_HTT[min_z:max_z, min_row:max_row, min_col:max_col]

        # Perform dilation iteratively on the cropped mask to avoid memory overload
        dilated_cropped_mask = cropped_mask.copy()
        for _ in range(border_width):
            dilated_cropped_mask = binary_dilation(dilated_cropped_mask)
        cropped_surrounding_mask = dilated_cropped_mask & ~cropped_mask
        if include_maskitself ==True:
            cropped_surrounding_mask = dilated_cropped_mask | cropped_mask

        # Get the surrounding values from the cropped image
        surrounding_values = cropped_image[cropped_surrounding_mask]

        # Calculate the median of the surrounding values
        if surrounding_values.size == 0:
            print('No surrounding values found, returning median as 0')
            median_surrounding_value = 0
        else:
            if use_mean:
                median_surrounding_value = np.mean(surrounding_values)
            else:

                median_surrounding_value = np.median(surrounding_values)
        return median_surrounding_value, cropped_surrounding_mask, (min_z, max_z, min_row, max_row, min_col, max_col)
    def analyze_label_intensities(self, filtered_labels, image_HTT):
        unique_labels = np.unique(filtered_labels)
        total_intensities = []
        centers_of_mass = []  # To store the center of mass for each cluster

        combined_surrounding_mask = np.zeros_like(filtered_labels, dtype=bool)
        def _calculate_surrounding_median_3D(image_HTT, mask, border_width=20):
            mask_indices = np.argwhere(mask)
            if mask_indices.size == 0:
                print('Empty mask, skipping...')
                #return 0  # Avoid processing empty masks

            min_z, min_row, min_col = np.min(mask_indices, axis=0)
            max_z, max_row, max_col = np.max(mask_indices, axis=0)

            # Expand the bounding box by border_width, ensuring it stays within image bounds
            min_z = max(0, min_z - border_width)
            max_z = min(image_HTT.shape[0], max_z + border_width + 1)
            min_row = max(0, min_row - border_width)
            max_row = min(image_HTT.shape[1], max_row + border_width + 1)
            min_col = max(0, min_col - border_width)
            max_col = min(image_HTT.shape[2], max_col + border_width + 1)

            # Crop the mask and image to the expanded bounding box
            cropped_mask = mask[min_z:max_z, min_row:max_row, min_col:max_col]
            cropped_image = image_HTT[min_z:max_z, min_row:max_row, min_col:max_col]

            # Perform dilation iteratively on the cropped mask to avoid memory overload
            dilated_cropped_mask = cropped_mask.copy()
            for _ in range(border_width):
                dilated_cropped_mask = binary_dilation(dilated_cropped_mask)
            cropped_surrounding_mask = dilated_cropped_mask & ~cropped_mask

            # Get the surrounding values from the cropped image
            surrounding_values = cropped_image[cropped_surrounding_mask]

            # Calculate the median of the surrounding values
            if surrounding_values.size == 0:
                print('No surrounding values found, returning median as 0')
                median_surrounding_value = 0
            else:
                median_surrounding_value = np.median(surrounding_values)
            return median_surrounding_value, cropped_surrounding_mask, (min_z,max_z, min_row,max_row, min_col,max_col)

        for label_idx in tqdm(unique_labels):
            if label_idx == 0:  # Skip the background
                continue

            mask = filtered_labels == label_idx
            mask_size = np.sum(mask)
            if mask_size == 0:
                print('Skipped label, should not be happening')
                continue  # Skip if mask is empty

            #print(f'\nSize of mask = {mask_size}')
            median_surrounding, cropped_surrounding_mask,(min_z,max_z, min_row,max_row, min_col,max_col) = _calculate_surrounding_median_3D(image_HTT, mask,
                                                                                                  border_width=10)
            # Map the cropped surrounding mask back to the original image
            if median_surrounding > 0:
                combined_surrounding_mask[min_z:max_z, min_row:max_row, min_col:max_col] |= cropped_surrounding_mask

            combined_surrounding_mask[min_z:max_z, min_row:max_row, min_col:max_col] |= cropped_surrounding_mask
            total_intensity = image_HTT[mask].sum() - mask_size * median_surrounding
            #print(f'\nLabel {label_idx} Intensity = {total_intensity}')
            total_intensities.append(total_intensity)
            com = center_of_mass(mask)  # returns (z, y, x) in floating-point
            centers_of_mass.append(com)

        #show_napari_images_lables(image_HTT, filtered_labels, combined_surrounding_mask)
        return np.array(total_intensities), np.array(centers_of_mass)

    def analyze_label_intensitiesv2(self, filtered_labels, image_HTT, border_width=10):
        """
        Analyze intensity and size of each label in a 3D image.

        Parameters
        ----------
        filtered_labels : ndarray
            3D label image where each cluster has a unique integer label
        image_HTT : ndarray
            3D intensity image (same shape as filtered_labels)
        border_width : int
            Width of the surrounding region for background estimation

        Returns
        -------
        total_intensities : ndarray
            Background-corrected total intensity for each label (in order of unique labels)
        centers_of_mass : ndarray
            Center of mass (z, y, x) for each label in global coordinates
        label_sizes : ndarray
            Number of voxels (3D volume) for each label - matches indices with intensities
        label_cvs : ndarray
            Coefficient of variation (std/mean) for each label's background-corrected intensities
        """
        unique_labels = np.unique(filtered_labels)
        total_intensities = []
        centers_of_mass = []
        label_sizes = []  # Track 3D volume (voxel count) for each label
        label_cvs = []  # Track coefficient of variation for each label

        # Get bounding boxes for each label (note: find_objects returns a list with index label-1)
        slices = ndimage.find_objects(filtered_labels)
        combined_surrounding_mask = np.zeros_like(filtered_labels, dtype=bool)

        for label in tqdm(unique_labels):
            if label == 0:  # Skip background
                continue
            # Get the bounding box from find_objects
            sl = slices[label - 1]  # find_objects returns list indexed by label-1
            if sl is None:
                # This should not happen if label exists, but check anyway
                continue

            # Get the slice indices for each dimension (z, y, x)
            z_sl, y_sl, x_sl = sl

            # Expand the bounding box by border_width, staying within image bounds
            min_z = max(0, z_sl.start - border_width)
            max_z = min(image_HTT.shape[0], z_sl.stop + border_width)
            min_row = max(0, y_sl.start - border_width)
            max_row = min(image_HTT.shape[1], y_sl.stop + border_width)
            min_col = max(0, x_sl.start - border_width)
            max_col = min(image_HTT.shape[2], x_sl.stop + border_width)

            # Create the mask for the current label within its bounding box
            cropped_label = filtered_labels[min_z:max_z, min_row:max_row, min_col:max_col]
            cropped_mask = cropped_label == label
            mask_size = np.count_nonzero(cropped_mask)
            if mask_size == 0:
                print(f"Skipped label {label}, mask size is 0")
                continue

            # Get the corresponding cropped image
            cropped_image = image_HTT[min_z:max_z, min_row:max_row, min_col:max_col]

            # Instead of iterating dilation, use iterations argument
            dilated_cropped_mask = binary_dilation(cropped_mask, iterations=border_width)
            cropped_surrounding_mask = dilated_cropped_mask & ~cropped_mask

            # Compute median intensity of the surrounding region
            surrounding_values = cropped_image[cropped_surrounding_mask]
            if surrounding_values.size == 0:
                median_surrounding = 0
                print(f'Label {label}: no surrounding values found, using median 0')
            else:
                median_surrounding = np.median(surrounding_values)

            # Update combined surrounding mask
            combined_surrounding_mask[min_z:max_z, min_row:max_row, min_col:max_col] |= cropped_surrounding_mask

            # Compute total intensity corrected by subtracting median-surrounding contribution
            total_intensity = cropped_image[cropped_mask].sum() - mask_size * median_surrounding
            total_intensities.append(total_intensity)

            # Store the 3D volume (voxel count) for this label
            label_sizes.append(mask_size)

            # Compute coefficient of variation (std/mean) for background-corrected intensities
            bg_corrected_values = cropped_image[cropped_mask] - median_surrounding
            label_mean = bg_corrected_values.mean()
            if label_mean > 0:
                label_cv = bg_corrected_values.std() / label_mean
            else:
                label_cv = 0.0
            label_cvs.append(label_cv)

            # Compute center of mass using SciPy's function
            com = center_of_mass(cropped_mask)
            # Translate the center-of-mass back to the coordinate system of the full image
            com_global = (com[0] + min_z, com[1] + min_row, com[2] + min_col)
            centers_of_mass.append(com_global)

        return np.array(total_intensities), np.array(centers_of_mass), np.array(label_sizes, dtype=np.int64), np.array(label_cvs)





    def gauss_mle_flex_sigma(self, smp_arr,lamda, initial_guess = None):
        bounds = [[1, self.roisize - 1],
                       [1, self.roisize - 1],
                       [0, 1e5],
                       [0, 1e6],
                        [0.5,2]]  # x, y, photons, bg

        if initial_guess is None:
            initial_arr = np.zeros((np.size(smp_arr, 0),5))
            initial_arr[:, :2] = np.array([self.roisize/2, self.roisize/2])  # position
            initial_arr[:, 2] = 1500  # photons
            initial_arr[:, 3] = 5  # bg
            initial_arr[:,4] = 1.0 # sigma

        else:
            initial_arr = np.zeros((np.size(smp_arr, 0), 4))
            initial_arr[:, :2] = np.array([initial_guess[0], initial_guess[1]])  # position
            initial_arr[:, 2] = initial_guess[2]  # photons
            initial_arr[:, 3] = initial_guess[3]  # bg
            initial_arr[:, 4] = initial_guess[4]  # bg

        model = Gaussian_flexsigma(self.roisize)

        param_range_ = torch.tensor(bounds).to(self.dev)
        initial_ = torch.tensor(initial_arr).to(self.dev)
        #smp_arr = np.cat(smp_arr)
        smp_ = torch.tensor(smp_arr).to(self.dev)

        mle = LM_MLE_with_iter(model, lambda_=lamda, iterations=self.iterations,
                               param_range_min_max=param_range_,
                               tol=torch.tensor([1e-3, 1e-3, 1e-2, 1e-2,1e-3]).to(self.dev))

        #mle = torch.jit.script(mle)



        params_, loglik_, traces_ = mle.forward(smp_.type(torch.float32),
                                               initial_.type(torch.float32)
                                               )
        mu,_ = model.forward(params_)
        params = params_.detach().cpu().numpy()
        iterations_vector = np.zeros(len(params), dtype=int)  # Simplify np.size(params, 0) to len(params)
        traces = traces_.detach().cpu().numpy()
        for loc in range(len(params)):
            try:
                # Find the first index where traces[:, loc, 1] equals 0 and store in iterations_vector
                iterations_vector[loc] = np.where(traces[:, loc, 1] == 0)[0][0]
            except IndexError:
                # If no such index, default to self.iterations
                iterations_vector[loc] = self.iterations

        # Now, filter params where iterations_vector equals self.iterations
        # Ensure iterations_vector is a 2D column vector for consistency or further operations
        iterations_vector = iterations_vector[:,
                            None]  # This line seems to be for reshaping, ensure iterations_vector is used appropriately afterwards

        # Initial filtering based on iteration criteria
        iteration_filtered_indices = iterations_vector.flatten() == self.iterations

        # Placeholder for bounds filtering
        bounds_filtered_indices = np.ones(len(params), dtype=bool)  # Initialize all True

        # Track how many parameters are filtered due to not meeting bound conditions
        num_filtered_by_bounds = 0

        # Apply bounds for each parameter across all spots
        for i, (lower_bound, upper_bound) in enumerate(bounds):
            within_bounds = (params[:, i] >= 1.1*lower_bound) & (params[:, i] <= 0.9*upper_bound)
            previously_within_bounds_count = np.sum(bounds_filtered_indices)
            bounds_filtered_indices &= within_bounds  # Update with current parameter bounds criteria
            currently_within_bounds_count = np.sum(bounds_filtered_indices)

            # Update count of parameters filtered by bounds based on this parameter
            num_filtered_by_bounds += (previously_within_bounds_count - currently_within_bounds_count)

        # Combine iteration and bounds criteria
        final_filtered_indices = (~iteration_filtered_indices) & bounds_filtered_indices

        # Filter params using the combined criteria
        final_filtered_params = params[final_filtered_indices]

        # filter mu and smp
        mu_fil = mu.detach().cpu().numpy()[final_filtered_indices]
        smp_fil = smp_.detach().cpu().numpy()[final_filtered_indices]
        # Results
        total_params = len(params)
        num_filtered_params = len(final_filtered_params)
        num_filtered_by_iterations = np.sum(~iteration_filtered_indices) - num_filtered_params

        print(f"Total number of parameters: {total_params}")
        print(f"Number of parameters filtered based on iterations: {num_filtered_by_iterations}")
        print(f"Number of parameters filtered based on bounds: {num_filtered_by_bounds}")
        print(f"Number of parameters filtered based on both criteria: {num_filtered_params}")

        return params_,mu_fil, smp_fil

    def process_labels(self,label_image, smooth_sigma=3):
        """
        Process the label image by filling holes and smoothing labels slice by slice.

        Parameters:
        - label_image (np.ndarray): The input 3D label image (integer labels).
        - smooth_sigma (float): Standard deviation for Gaussian smoothing. If 0, Gaussian smoothing is skipped.

        Returns:
        - np.ndarray: The processed label image with original labels preserved.
        """
        # Initialize an empty array for processed labels
        processed_label = np.zeros_like(label_image, dtype=label_image.dtype)

        # Process each 2D slice
        print("Processing slices...")
        for i in tqdm(range(label_image.shape[0])):  # Assuming depth is the first dimension
            slice_image = label_image[i, :, :]

            # Ensure all labeled regions are correctly represented
            binary_slice = slice_image > 0

            # Optionally apply Gaussian smoothing
            if smooth_sigma > 0:
                smoothed_slice = gaussian_filter(binary_slice.astype(float), sigma=smooth_sigma)
                smoothed_binary_slice = smoothed_slice > 0.5  # Convert back to binary
            else:
                smoothed_binary_slice = binary_slice

            # Fill holes in the smoothed binary slice
            filled_binary_slice = binary_fill_holes(smoothed_binary_slice)


            processed_slice = filled_binary_slice.copy()

            # Assign the processed slice back to the processed_label array
            processed_label[i, :, :] = processed_slice
        labeled_image, _ = ndi_label(processed_label)

        return labeled_image
    def generate_label(self, min_size, max_size, mode='blue', image_=None):
        """
        Generates a label mask (or segmentation) for the specified mode.

        :param min_size: Minimum size for connected components (e.g. nuclear segmentation).
        :param max_size: Maximum size for connected components.
        :param mode: String key identifying which channel/config to use.
                     e.g. 'blue', 'green', 'orange', 'red', etc.
        :param image_: Optionally pass a pre-loaded image; if None, we'll load it from disk.
        :return: (label_mask, image_data)
        """
        # 1) Get the channel index from self.channel_map
        if mode not in self.channel_map:
            raise ValueError(f"Mode '{mode}' not found in channel_map. "
                             f"Available modes: {list(self.channel_map.keys())}")

        channel_idx = self.channel_map[mode]

        # 2) Get the path to the YAML or config file from self.config_paths, if applicable
        #    (If you have no config for a particular mode, decide how to handle it.)
        config_path = self.config_paths.get(mode, None)
        if config_path is None:
            raise ValueError(f"No config_paths entry for mode '{mode}'. "
                             f"Available config paths: {list(self.config_paths.keys())}")

        # 3) Load the image if none was provided
        if image_ is None:
            image_data,_ = self.load_image(self.image_file, channel_to_load=channel_idx)
        else:
            image_data = image_
        label_image = function_htt(image_data, config_path)
        # show_napari(label_image)
        # show_napari(image)
        print('filter labels')
        filtered_labels = self.filter_labels_by_size(label_image, min_size, max_size)
        if mode == 'dapi':
            filtered_labels = self.process_labels(filtered_labels[0], smooth_sigma=10)
        else:
            filtered_labels = self.process_labels(filtered_labels[0], smooth_sigma=3)

        filtered_labels = filtered_labels[None,...]

        filtered_labels = self.filter_labels_by_size(filtered_labels, min_size, max_size)

        print('done filter labels')

        # Relabel the filtered labels to ensure they run from 1 to n
        relabeled_filtered_labels = label(filtered_labels[0, ...], background=0)
        if self.generate_plots:
            label_sizes = np.bincount(relabeled_filtered_labels.ravel())[1::]
            plt.hist(label_sizes,bins=80)
            plt.xlabel('Label size after filtering')
            plt.ylabel('Probability')
            plt.show()
            label_sizes = np.bincount(label_image.ravel())[1::]
            plt.hist(label_sizes,bins=80)
            plt.xlabel('Label size before filtering')
            plt.ylabel('Probability')
            plt.show()
        label_sizes = np.bincount(relabeled_filtered_labels.ravel())[1::]
        return relabeled_filtered_labels, image_data, label_sizes


    def detect_and_fit(self, detection_cfg, fit_cfg, image_HTT,cfg_color, batch_size=1000, label_mask=None,
                      fit_bg_per_slice=True, fit_sigma=False):
        print('detect and fit...')


        #mip_image = self.load_image(self.image_file).astype(np.float32)[detection_cfg['channel'], ...]
        damping_lm = fit_cfg['damping_factor']
        uniform_filter1_size = detection_cfg['uniform_filter1_size']
        uniform_filter2_size = detection_cfg['uniform_filter2_size']
        local_max_filter_size = detection_cfg['local_max_filter_size']
        detection_threshold = cfg_color['intensity_threshold']
        estim_sigma = cfg_color['sigma']
        roisize = detection_cfg['roisize']
        min_distance =detection_cfg['min_distance']
        image_HTT_corrected = self.convert_to_photons(image_HTT)
        image_HTT_corrected_ = torch.tensor(image_HTT_corrected, dtype=torch.float32).to(self.dev)
        mip_tensor= torch.max(image_HTT_corrected_, dim=0).values
        print('detect and fit...')


        detector = SMLMSpotDetector(
            uniform_filter1_size,
            uniform_filter2_size,
            local_max_filter_size,
            detection_threshold,
            roisize,
            min_distance
        )
        print('detection...')
        if label_mask is None:
            filtered_rois, filtered_coords= detector.detect(mip_tensor)
        else:

            rois, filtered_coords = detector.detect(mip_tensor)
            print('done detection...')

            mip_labels = (np.max(label_mask, 0) > 0.5)
            print('filter rois...')

        #show_tensor_spots(mip_tensor, filtered_coords)
        print('extract rois...')
        zslices = fit_cfg['zslices']
        rois_3d, z_starts = self.extract_3d_rois(image_HTT_corrected, filtered_coords, detection_cfg['roisize'], zslices)
        #show_napari(rois_3d)
        print(' done extract rois...')
        smp_ = torch.tensor(rois_3d, dtype=torch.float32).to(self.dev)
        print(' calculate COM')
        #show_tensor(smp_)
        if smp_.nelement() == 0:
            # Handle the case where smp_ is empty
            print("Input tensor smp_ is empty.")
            return None, None, None, None, None, None, None,None,None
        else:
            difff = int((detection_cfg['roisize'] - self.roisize)//2)
            z_estimate = torch.argmax(torch.sum(smp_, dim=(-1, -2)), dim=-1)
            flattened = smp_[:,:,difff: detection_cfg['roisize']-difff: detection_cfg['roisize']-difff].flatten(start_dim=-3)  # Equivalent to flatten(-3)

            # Step 2: Select the lowest 5 pixels using torch.topk
            # torch.topk returns the top 'k' elements; with largest=False, it returns the smallest 'k'

            k = int(self.roisize*self.roisize*zslices)
            lowest_5_pixels = torch.topk(flattened, k=5, dim=-1, largest=False).values
            # Shape of lowest_5_pixels: (Batch, Channels, 5)

            # Step 3: Compute the median of the lowest 5 pixels
            # The median of 5 elements is the 3rd smallest value

            #test,_ = model.forward(initial)
            #show_tensor((torch.concatenate((smp_,test),dim=-1)))
            if fit_sigma is not False:
                bounds_mle = fit_cfg['bounds_mle_sigma']
            else:
                bounds_mle = fit_cfg['bounds_mle']
            bg_bound = bounds_mle[4]

            # Build new_bounds:
            # First four parameters remain unchanged: [x, y, z, N]


            bg_init = torch.median(lowest_5_pixels)
            photons_init = torch.clamp(
                torch.sum(smp_, dim=[-3, -2, -1]) - bg_init * self.roisize ** 2 * smp_.size(1),
                min=1)

            initial = torch.zeros((np.size(smp_, 0), 5)).to(self.dev)
            initial[:, :2] = torch.tensor([self.roisize / 2, self.roisize / 2]).to(self.dev)
            initial[:, 2] = z_estimate.to(self.dev)
            initial[:, 3] = photons_init
            initial[:, 4] = bg_init[..., None]
            new_bounds = bounds_mle
            # Convert to tensor and move to the device.
            param_range = torch.tensor(new_bounds, device=self.dev)

            if estim_sigma is None or fit_sigma is not False:
                initial_sig = torch.zeros((smp_.size(0), initial.size(-1)+3)).to(self.dev)
                initial_sig[:, :initial.size(-1)] = initial
                initial_sig[:, initial.size(-1)] = fit_cfg['initial_sigma'][0]
                initial_sig[:, initial.size(-1)+1] = fit_cfg['initial_sigma'][1]
                initial_sig[:, initial.size(-1)+2] = fit_cfg['initial_sigma'][2]
                initial = initial_sig * 1
                estim_sigma =None



            model = Gaussian3DPSF(self.roisize, zslices, estim_sigma,fit_bg_per_slice=fit_bg_per_slice)
            iterations = fit_cfg['iterations']
            # Ensure batch_size is valid
            if batch_size > smp_.size(0):
                batch_size = smp_.size(0)
            smp_chunks = torch.chunk(smp_, smp_.size(0) // batch_size, dim=0)
            initial_chunks = torch.chunk(initial, initial.size(0) // batch_size, dim=0)
            mu_all_batches = []
            params_all_batches = []
            traces_all_batches = []
            pfa_around_pos_l_all_batches = []
            # Iterate over chunks
            torch.cuda.empty_cache()
            # 🔻 NEW: timing + tqdm wrapper ---------------------------------------------
            num_batches = len(smp_chunks)  # assumes smp_chunks and initial_chunks same length
            import time
            loop_start = time.perf_counter()

            for idx, (smp_batch, initial_batch) in enumerate(
                    tqdm(zip(smp_chunks, initial_chunks),
                         total=num_batches, desc="Fitting batches"), 1):
                batch_start = time.perf_counter()
                print(f"\nBatch {idx}/{num_batches} – starting estimation…")

                # --- main work ----------------------------------------------------------
                params_batch_, traces_batch_, pfa_around_pos_l, bg_estim3D_resized = LM_MLE_for_zstack(
                    model,
                    smp_batch.to(torch.float32),
                    param_range,
                    initial_batch,
                    1e-2,
                    self.dev,
                    iterations,
                    damping_lm=damping_lm,
                    roisize=self.roisize,
                )

                mu_batch, jac_batch = model.forward(params_batch_, bg_constant=bg_estim3D_resized)

                mu_all_batches.append(mu_batch)
                params_all_batches.append(params_batch_)
                traces_all_batches.append(traces_batch_)
                pfa_around_pos_l_all_batches.append(pfa_around_pos_l)
                # -----------------------------------------------------------------------

                batch_secs = time.perf_counter() - batch_start
                print(f"Batch {idx} finished in {batch_secs:,.2f} s")

            total_secs = time.perf_counter() - loop_start
            print(f"\n✅ All {num_batches} batches processed in {total_secs:,.2f} s")
            # After processing all batches, concatenate the results to get the final tensors
            mu = torch.cat(mu_all_batches, dim=0)  # Concatenate along the batch dimension
            params_ = torch.cat(params_all_batches, dim=0)
            traces_ = torch.cat(traces_all_batches, dim=1)
            pfa_around_pos_l=np.concatenate(pfa_around_pos_l_all_batches, axis=0)
            # show_tensor((torch.concatenate((smp_, mu), dim=-1)))
            params = params_.detach().cpu().numpy()

            iterations_vector = np.zeros(len(params), dtype=int)
            traces = traces_.detach().cpu().numpy()
            test_traces = torch.permute(traces_, (1,2,0)).detach().cpu().numpy()
            for loc in range(len(params)):
                try:
                    iterations_vector[loc] = np.where(traces[:, loc, 1] == 0)[0][0]
                except IndexError:
                    iterations_vector[loc] = iterations

            iterations_vector = iterations_vector[:, None]
            iteration_filtered_indices = iterations_vector.flatten() == iterations
            bounds_filtered_indices = np.ones(len(params), dtype=bool)
            num_filtered_by_bounds = 0
            for i, (lower_bound, upper_bound) in enumerate(new_bounds):
                within_bounds = (params[:, i] >= 1.1 * lower_bound) & (params[:, i] <= 0.9 * upper_bound)
                previously_within_bounds_count = np.sum(bounds_filtered_indices)
                bounds_filtered_indices &= within_bounds
                currently_within_bounds_count = np.sum(bounds_filtered_indices)
                num_filtered_by_bounds += (previously_within_bounds_count - currently_within_bounds_count)

            final_filtered_indices = (~iteration_filtered_indices) & bounds_filtered_indices
            if estim_sigma is None:
                final_pfa = []
            else:
                final_pfa = pfa_around_pos_l
            final_filtered_params = params[final_filtered_indices]

            mu_fil = mu.detach().cpu().numpy()[final_filtered_indices]
            smp_fil = smp_.detach().cpu().numpy()[final_filtered_indices]

            total_params = len(params)
            num_filtered_params = len(final_filtered_params)
            num_filtered_by_iterations = np.sum(~iteration_filtered_indices) - num_filtered_params

            print(f"Total number of parameters: {total_params}")
            print(f"Number of parameters filtered based on iterations: {num_filtered_by_iterations}")
            print(f"Number of parameters filtered based on bounds: {num_filtered_by_bounds}")
            print(f"Number of parameters left: {num_filtered_params}")

            traces = torch.permute(traces_, (1, 0, 2)).detach().cpu().numpy()
            # chi error
            torch.cuda.empty_cache()
            return (mu_fil, smp_fil, final_filtered_params, traces, mip_tensor.detach().cpu().numpy(),
                    filtered_coords, z_starts, final_filtered_indices,final_pfa, params)

    def remove_labels_touching_spots(self,
                                     coords,
                                     label_stack,  # MUST be 3D: (Z,H,W)
                                     dilation_radius=0,
                                     mip_strategy="first",  # "first" or "last" non-zero along Z
                                     connectivity=1):
        """
        Remove entire labels (set to 0) if any (dilated) spot overlaps them.
        Always returns: coords_out, pruned_mip (2D), pruned_stack (3D).

        - coords: array-like of (y,x) or (z,y,x)
        - label_stack: 3D boolean or integer-labeled array (Z,H,W)
            * If boolean, we 3D-label it first to get integer IDs.
            * If integer, we assume 3D-connected components already labeled.
        - dilation_radius: 2D disk radius (pixels) applied per-slice around spots.
        """

        label_stack = np.asarray(label_stack)
        assert label_stack.ndim == 3, "label_stack must be 3D (Z,H,W)"
        Z, H, W = label_stack.shape
        is_bool = (label_stack.dtype == bool)

        # ---- Ensure integer labels in 3D (so 'IDs' exist to remove) ------------------------
        if is_bool:
            lab3d = cc_label(label_stack.astype(np.uint8), connectivity=connectivity)
        else:
            lab3d = label_stack.copy()

        # ---- Build 3D spot mask ------------------------------------------------------------
        coords = np.asarray(coords)
        spot3d = np.zeros((Z, H, W), dtype=bool)

        if coords.size > 0:
            if coords.ndim == 1:
                coords = coords[None, :]
            if coords.shape[1] == 3:
                zz = np.clip(np.rint(coords[:, 0]).astype(int), 0, Z - 1)
                yy = np.clip(np.rint(coords[:, 1]).astype(int), 0, H - 1)
                xx = np.clip(np.rint(coords[:, 2]).astype(int), 0, W - 1)
                spot3d[zz, yy, xx] = True
            else:
                yy = np.clip(np.rint(coords[:, 0]).astype(int), 0, H - 1)
                xx = np.clip(np.rint(coords[:, 1]).astype(int), 0, W - 1)
                spot3d[:, yy, xx] = True  # no z → mark across all slices

        # ---- Dilate *per slice* with a 2D disk --------------------------------------------
        selem = disk(max(0, int(dilation_radius)))
        if selem.size > 1:
            for z in range(Z):
                if spot3d[z].any():
                    spot3d[z] = binary_dilation(spot3d[z], selem)

        # ---- Collect label IDs to remove & zero them out -----------------------------------
        ids_to_remove = np.unique(lab3d[spot3d])
        ids_to_remove = ids_to_remove[ids_to_remove != 0]  # drop background

        pruned_stack = lab3d.copy()
        if ids_to_remove.size:
            pruned_stack[np.isin(pruned_stack, ids_to_remove)] = 0

        # ---- Make a 2D labeled MIP that preserves numbering --------------------------------
        def _label_mip_with_ids(stack, strategy="first"):
            # stack is integer-labeled (Z,H,W)
            nz = stack > 0
            has = nz.any(axis=0)  # (H,W)
            if strategy == "last":
                z_idx = (stack.shape[0] - 1) - np.argmax(nz[::-1], axis=0)
            else:  # "first"
                z_idx = np.argmax(nz, axis=0)
            mip = np.zeros((H, W), dtype=stack.dtype)
            yy, xx = np.where(has)
            mip[yy, xx] = stack[z_idx[yy, xx], yy, xx]
            return mip

        pruned_mip = _label_mip_with_ids(pruned_stack, strategy=mip_strategy)

        return np.asarray(coords), pruned_mip, pruned_stack

        return np.asarray(coords), pruned_mip
    def filter_rois_by_labels(self, rois, coords, label_mask, dilation_radius):
        selem = disk(dilation_radius)
        print(np.shape(label_mask))
        print(np.shape(selem))
        dilated_label_mask = binary_dilation(label_mask, selem)
        filtered_rois = []
        filtered_coords = []
        for roi, (y, x) in zip(rois, coords):
            if not dilated_label_mask[y, x]:
                filtered_rois.append(roi)
                filtered_coords.append((y, x))
        return np.array(filtered_rois), np.array(filtered_coords), dilated_label_mask

    def calculate_centers_of_mass(self, rois):
        centers_of_mass = [center_of_mass(roi) for roi in rois]
        return np.array(centers_of_mass)

    def compute_distance_to_closest_dapi(self,dapi_labels, z_starts, filt_indices, final_params, com_array, sampling=None):
        """
        Compute the signed distance from given coordinates to the nearest DAPI border.
        Distances are computed in physical units using provided sampling:
          - z: 0.5 µm (by default)
          - x,y: 0.1625 µm (by default)

        Parameters:
        -----------
        dapi_labels : ndarray
            A 3D labeled mask where nonzero values indicate DAPI regions.
        z_starts : array-like or None
            Array used to compute the z component if com_array is 2D. (Ignored if com_array has 3 columns.)
        filt_indices : array-like or None
            Indices to filter z_starts and final_params when computing the z component.
        final_params : ndarray or None
            Array from which the z offset (assumed to be in column index 3) is extracted.
        com_array : ndarray or None
            Array of coordinates. If it has 3 columns, they are assumed to be (z, x, y).
            If it has 2 columns (x, y), the z coordinate is computed using z_starts and final_params.
        sampling : tuple or None
            Physical spacing for each axis in (z, x, y) order. Default is (0.5, 0.1625, 0.1625).

        Returns:
        --------
        distance_to_closest_dapi : ndarray or None
            Array of signed distances at each coordinate. Negative values indicate points inside
            the DAPI regions, and positive values indicate points outside.
            Returns None if required inputs are missing or empty.
        """
        if sampling is None:
            sampling = (0.5, 0.1625, 0.1625)

        # Validate dapi_labels
        if dapi_labels is None or not dapi_labels.size:
            print("dapi_labels is None or empty. Returning None.")
            return None

        try:
            # Create a binary mask from dapi_labels (nonzero values indicate DAPI regions)
            dapi_mask = dapi_labels > 0

            # Compute distance transforms using the specified physical spacing
            dist_inside = distance_transform_edt(dapi_mask, sampling=sampling)
            dist_outside = distance_transform_edt(~dapi_mask, sampling=sampling)

            # Signed distance: negative inside DAPI regions, positive outside
            signed_distance = np.where(dapi_mask, -dist_inside, dist_outside)
        except Exception as e:
            print("Error during distance transform computation:", e)
            return None

        try:
            # Determine the coordinate array
            if com_array is not None:
                if com_array.shape[1] == 3:
                    # Use provided (z, x, y) coordinates directly
                    coords = com_array
                elif com_array.shape[1] == 2:
                    # Need to compute the z component from z_starts and final_params
                    if z_starts is None or filt_indices is None or final_params is None:
                        print(
                            "Missing z_starts, filt_indices, or final_params for computing z component. Returning None.")
                        return None
                    if final_params.shape[1] < 4:
                        print("final_params does not have at least 4 columns. Returning None.")
                        return None
                    # Compute z component and concatenate with (x, y)
                    z_component = (z_starts[filt_indices] + final_params[:, 3])[..., None]
                    coords = np.concatenate((z_component, com_array[filt_indices, :]), axis=-1)
                else:
                    print("com_array must have either 2 or 3 columns. Returning None.")
                    return None
            else:
                print("com_array is None. Returning None.")
                return None

            # Check that the coordinates array is not empty.
            if coords.size == 0:
                print("Computed coordinates array is empty. Returning None.")
                return None
        except Exception as e:
            print("Error computing coordinates:", e)
            return None

        try:
            # Interpolate the signed distance at each coordinate.
            # map_coordinates expects coordinates in (z, x, y) order as separate arrays,
            # so we pass coords.T (each row corresponds to an axis).
            distance_to_closest_dapi = map_coordinates(signed_distance, coords.T, order=1, mode='nearest')
        except Exception as e:
            print("Error during interpolation of signed distances:", e)
            return None

        return distance_to_closest_dapi
    def extract_3d_rois(self, image, coords, roisize, zslices, sigma=1):
        """
        Extracts 3D ROIs from a smoothed image based on provided coordinates.

        Parameters:
            image (numpy.ndarray): The input 3D image array with shape (Z, Y, X).
            coords (list or array): List of (y, x) tuples indicating ROI centers.
            roisize (int): Size of the ROI in the Y and X dimensions.
            zslices (int): Number of slices in the Z dimension for each ROI.
            sigma (float or sequence, optional): Standard deviation for Gaussian kernel.

        Returns:
            tuple: A tuple containing:
                - rois (numpy.ndarray): Extracted ROIs with shape (N, zslices, roisize, roisize).
                - z_starts (numpy.ndarray): Starting Z indices for each ROI.
        """
        # Apply Gaussian smoothing to the entire image
        smoothed_image = gaussian_filter(image, sigma=sigma)

        half_roisize = roisize // 2
        half_zslices = zslices // 2
        rois = []
        z_starts = []

        for idx, (y, x) in enumerate(coords):
            # Determine if roisize is odd or even
            is_odd = roisize % 2 != 0

            # Calculate y_min and y_max
            y_min = max(y - half_roisize, 0)
            y_max = y + half_roisize + 1 if is_odd else y + half_roisize

            # Calculate x_min and x_max
            x_min = max(x - half_roisize, 0)
            x_max = x + half_roisize + 1 if is_odd else x + half_roisize

            # Adjust y_max and x_max if they exceed image boundaries
            y_max = min(y_max, image.shape[1])
            x_max = min(x_max, image.shape[2])

            # Extract the local ROI from the smoothed image
            local_roi = smoothed_image[:, y_min:y_max, x_min:x_max]

            # Check if the local ROI has the expected size
            expected_y_size = roisize if y_max - y_min == roisize else y_max - y_min
            expected_x_size = roisize if x_max - x_min == roisize else x_max - x_min

            if local_roi.shape[1] != expected_y_size or local_roi.shape[2] != expected_x_size:
                print(
                    f"Warning: ROI at (x={x}, y={y}) has inconsistent local size "
                    f"({local_roi.shape[1]}, {local_roi.shape[2]}), expected ({expected_y_size}, {expected_x_size}). "
                    "This may affect center of mass calculation."
                )

            # Calculate the center of mass in the z-dimension for the ROI
            #z_com = int(np.round(center_of_mass(local_roi))[0])

            # Ensure z_com is within valid bounds
            test = np.sum(local_roi,axis=(-1,-2))
            z_com = np.argmax(test)
            z_com = max(min(z_com, image.shape[0] - 1), 0)

            # Determine the start and end slices in the z-dimension
            z_start = max(z_com - half_zslices, 0)
            z_end = z_start + zslices

            # Adjust z_end if it exceeds the image bounds
            if z_end > image.shape[0]:
                z_end = image.shape[0]
                z_start = max(z_end - zslices, 0)

            # Extract the ROI from the smoothed image for consistency
            roi = image[z_start:z_end, y_min:y_max, x_min:x_max]

            # Final check to ensure ROI has the expected shape
            if roi.shape == (zslices, roisize, roisize):
                rois.append(roi)
                z_starts.append(z_start)
            else:
                # Print a warning message and skip this ROI if the shape is not as expected
                print(
                    f"Skipping ROI at (x={x}, y={y}) with shape {roi.shape}, expected ({zslices}, {roisize}, {roisize})"
                )

        return np.array(rois), np.array(z_starts)
