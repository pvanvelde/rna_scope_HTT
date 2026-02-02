import torch
import torch.nn.functional as F
import numpy as np
import tifffile
import os
from tqdm import tqdm
import glob

def show_napari(image):
    import napari
    with napari.gui_qt():
        viewer = napari.view_image(image, title="Detected ROIs")

def show_napari_spots(image, spots):
    import napari
    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(image, name="Summed Images")
        points = np.array([[y, x ] for y, x in spots])
        viewer.add_points(points, edge_color='red', face_color='transparent', name="spots")

class SMLMSpotDetector:
    def __init__(self, uniform_filter1_size, uniform_filter2_size, local_max_filter_size, intensity_threshold, roisize, min_distance):
        self.uniform_filter1_size = uniform_filter1_size
        self.uniform_filter2_size = uniform_filter2_size
        self.local_max_filter_size = local_max_filter_size
        self.intensity_threshold = intensity_threshold
        self.roisize = roisize
        self.min_distance = min_distance

    def uniform_filter_2d(self, img, kernel_size):
        pad = kernel_size // 2
        img_padded = F.pad(img.unsqueeze(0).unsqueeze(0).float(), (pad, pad, pad, pad), mode='reflect')
        kernel = torch.ones((1, 1, kernel_size, kernel_size), device=img.device, dtype=torch.float32) / (kernel_size * kernel_size)
        filtered_img = F.conv2d(img_padded, kernel).squeeze()
        return filtered_img

    def comparison_filter_2d(self, img, kernel_size):
        pad = kernel_size // 2
        img_padded = F.pad(img.unsqueeze(0).unsqueeze(0).float(), (pad, pad, pad, pad), mode='reflect')
        pooled = F.max_pool2d(img_padded, kernel_size, stride=1, padding=0).squeeze()
        return pooled

    def filter_positions_pytorch(self,pos, min_distance_det):
        """
        Filters positions based on minimum distance using PyTorch.

        Args:
            pos (torch.Tensor): Tensor of shape (N, D), where N is the number of points
                                and D is the dimensionality (including any ID columns).
            min_distance_det (float): The minimum distance threshold.

        Returns:
            filtered_positions (torch.Tensor): Tensor containing positions that are
                                              close to at least one other point.
            isolated_positions (torch.Tensor): Tensor containing positions that are
                                              isolated (no other points within min_distance_det).
        """
        if pos.numel() == 0:
            return pos, pos

        # Extract coordinates, assuming the first column is an ID
        coords = pos[:, 1:].float()

        # Compute the pairwise distance matrix
        # Shape: (N, N)
        dist_matrix = torch.cdist(coords, coords, p=2)

        # Create a boolean mask where distance < min_distance_det
        # Exclude self-distances by setting the diagonal to False
        mask = dist_matrix < min_distance_det
        mask.fill_diagonal_(False)

        # Find indices where mask is True
        close_pairs = torch.nonzero(mask, as_tuple=False)

        if close_pairs.numel() == 0:
            # No close pairs found; all points are isolated
            return pos, pos

        # Extract unique indices of spots that are close to at least one other spot
        unique_indices = torch.unique(close_pairs.flatten())

        # Filter the final positions by the unique indices
        filtered_positions = pos[unique_indices]

        # Find indices of isolated spots
        N = pos.size(0)
        all_indices = torch.arange(N, device=pos.device)
        isolated_mask = ~torch.zeros(N, dtype=torch.bool, device=pos.device).scatter_(0, unique_indices, True)
        isolated_indices = all_indices[isolated_mask]

        # Filter the final positions by the isolated indices
        isolated_positions = pos[isolated_indices]

        return filtered_positions, isolated_positions
    def filter_by_distance(self, candidates, min_distance):
        if len(candidates) == 0:
            return candidates
        filtered_candidates = []
        for cand in candidates:
            cand = cand.float()  # Ensure the candidate is a float tensor
            if all(torch.dist(cand, f.float()) >= min_distance for f in filtered_candidates):
                filtered_candidates.append(cand)
        return torch.stack(filtered_candidates) if filtered_candidates else torch.empty((0, 2))



    def filter_positions_pytorch_gpu(self,coords, min_distance_det):

        isolated_positions_found = True

        # Compute pairwise distances on GPU
        dist_matrix = torch.cdist(coords.to(float), coords.to(float), p=2)

        # Rest of the processing remains the same
        mask = dist_matrix < min_distance_det
        mask.fill_diagonal_(False)

        close_pairs = torch.nonzero(mask, as_tuple=False)


        unique_indices = torch.unique(close_pairs.flatten())

        filtered_positions = coords[unique_indices]

        N = coords.size(0)
        all_indices = torch.arange(N, device=coords.device)
        isolated_mask = ~torch.zeros(N, dtype=torch.bool, device=coords.device).scatter_(0, unique_indices, True)
        isolated_indices = all_indices[isolated_mask]

        if len(isolated_indices) == 0:
            isolated_positions_found = False
        else:
            isolated_positions = coords[isolated_indices]

        return isolated_positions if isolated_positions_found else torch.empty((0, 2))

    def detect(self, image, background_image=None, max_spots=6000):


        if background_image is not None:
            filter_src = image - background_image
        else:
            filter_src = image
        print('filter')
        filtered1 = self.uniform_filter_2d(filter_src, self.uniform_filter1_size)
        filtered2 = self.uniform_filter_2d(filter_src, self.uniform_filter2_size)

        # Ensure the dimensions match
        min_h = min(filtered1.shape[0], filtered2.shape[0])
        min_w = min(filtered1.shape[1], filtered2.shape[1])
        filtered1 = filtered1[:min_h, :min_w]
        filtered2 = filtered2[:min_h, :min_w]

        filtered_diff = filtered1 - filtered2
        print('comparison filter')
        local_max_filtered = self.comparison_filter_2d(filtered_diff, self.local_max_filter_size)

        is_local_max = (filtered_diff == local_max_filtered)
        intensity_mask = (filtered_diff > self.intensity_threshold)
        # show_napari(filtered_diff.detach().cpu().numpy())
        # show_napari(image.detach().cpu().numpy())

        candidates = (is_local_max & intensity_mask).nonzero(as_tuple=False)


        print('filter by distance')
        # Filter candidates by minimum distance
        #candidates = self.filter_by_distance(candidates, self.min_distance)
        #test1 = self.filter_by_distance(candidates[0:1000,:], self.min_distance)
        #test2 = self.filter_by_distance_iterative(candidates[0:1000,:], self.min_distance)
        candidates = self.filter_positions_pytorch_gpu(candidates, self.min_distance)

        if len(candidates)>max_spots:
            scores = filtered_diff[candidates[:, 0], candidates[:, 1]]

            scores_sorted, indices = torch.sort(scores, descending=True)
            candidates = candidates[indices[0:max_spots]]
            scores_up = scores[indices[0:max_spots]]

        rois = []
        coords = []
        half_roisize = self.roisize // 2
        print('apeend list')
        for candidate in candidates:
            y, x = candidate.int()  # Convert tensors to integers
            if y >= half_roisize and y < image.shape[0] - half_roisize and x >= half_roisize and x < image.shape[1] - half_roisize:
                roi = image[y - half_roisize:y + half_roisize, x - half_roisize:x + half_roisize].cpu().numpy()
                rois.append(roi)
                coords.append([y.item(), x.item()])



        return np.array(rois), np.array(coords)

if __name__ == "__main__":
    uniform_filter1_size = 6
    uniform_filter2_size = 12
    local_max_filter_size = 9
    intensity_threshold = 20  # Adjust threshold based on dataset
    roisize = 16
    min_distance = 10
    channel = 2
    background = 100
    gain = 0.3
    detector = SMLMSpotDetector(uniform_filter1_size, uniform_filter2_size, local_max_filter_size, intensity_threshold, roisize, min_distance)

    # Load the image stack and compute the max intensity projection
    stack_path = '/media/pieter/Extreme SSD/Dropbox (UMass Medical School)/Huntingtin_RNAscope/HTT_Images_07_01_2024/c128-3/grunwald test_2024-07-01_17.58.50.tif'
    image_stack = tifffile.imread(stack_path).astype(np.float32)[channel, ...]
    mip_image = (np.max(image_stack, axis=0) - background) * gain

    # Convert MIP to tensor
    mip_tensor = torch.tensor(mip_image, device='cuda')

    # Run the detector and save results
    detector.detect(mip_tensor, background_image=None)

    base_path = os.path.dirname(stack_path)
    output_dir = os.path.join(base_path, "results")

    # Load and visualize the results
    rois = np.load(os.path.join(output_dir, "rois.npy"))
    coords = np.load(os.path.join(output_dir, "coords.npy"))
    summed_images = np.load(os.path.join(output_dir, "summed_images.npy"))
    #
    # show_napari(summed_images)
    show_napari_spots(summed_images, coords)