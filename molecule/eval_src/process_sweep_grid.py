import matplotlib.pyplot as plt
import os
import glob
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from pathlib import Path
from PIL import Image, ImageDraw
import copy
import numpy as np
import matplotlib.gridspec as gridspec
import torch

image_dir = "./eval_src/visualize_results/conditional_sweep_2024-09-23_06-16-03"
calculate_real_values = True
property = "alpha"
class_dir = "checkpoints/classifiers_ckpts/exp_class_alpha"
device = "cuda"

def once(image_dir, calculate_real_values, property, class_dir, device):
    # Get all png images from the directory, assuming filenames are ordered
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*[0-9][0-9][0-9].png")))

    # Corresponding values to place under each image
    values = []
    print(Path(image_dir))
    with open(Path(image_dir) / "property_values.log", "r") as f:
        lines = f.readlines()
        for line in lines:
            values.append(float(line.strip()))

    real_values = None
    if calculate_real_values:
        import sys
        sys.path.append(".")
        from omegaconf import OmegaConf
        from qm9.models import get_model, DistributionProperty
        from qm9.property_prediction import prop_utils
        from eval_src.eval_conditional_qm9 import get_classifier, get_dataloader
        from qm9.utils import compute_mean_mad

        dtype = torch.float32

        def get_real_values(image_dir):
            xyz_paths = sorted(glob.glob(os.path.join(image_dir, "*[0-9][0-9][0-9].txt")))

            classifier = get_classifier(class_dir).to(device)
            # Get generator and dataloader used to train the generator and evaluate the classifier
            dataset_args = {
                "dataset": "qm9_second_half",
                "conditioning": [property],
                "include_charges": True,
                "world_size": 1,
                "rank": 0,
                "filter_n_atoms": None,
                "remove_h": False,
                "batch_size": 128,
                "num_workers": 4,
                "datadir": "./data"
            }
            dataset_args = OmegaConf.create(dataset_args)

            dataloaders = get_dataloader(dataset_args)
            prop_dist = DistributionProperty(dataloaders['train'], [property])
            property_norms = compute_mean_mad(dataloaders, [property], dataset_args.dataset)
            prop_dist.set_normalizer(property_norms)

            # Create a dataloader with the generator
            mean, mad = property_norms[property]['mean'], property_norms[property]['mad']
            atom_type2one_hot_index = {
                "H": 0,
                "C": 1,
                "N": 2,
                "O": 3,
                "F": 4,
            }

            # Initialize lists to hold the data
            all_molecules = []
            max_molecule_size = 0

            # Read all xyz files and extract atom types and coordinates
            for path in xyz_paths:
                with open(path, 'r') as f:
                    lines = f.readlines()
                    num_atoms = int(lines[0].strip())
                    molecule_data = []

                    for line in lines[2:2 + num_atoms]:
                        parts = line.split()
                        atom_type = int(atom_type2one_hot_index[parts[0]])  # Assuming the first part is the atomic number
                        coordinates = list(map(float, parts[1:4]))  # x, y, z coordinates
                        one_hot = np.zeros(len(atom_type2one_hot_index))
                        one_hot[atom_type] = 1
                        molecule_data.append((one_hot, coordinates))

                    all_molecules.append(molecule_data)
                    max_molecule_size = max(max_molecule_size, num_atoms)

            # Prepare tensors: one_hot, x, node_mask
            num_molecules = len(all_molecules)
            one_hot_tensor = np.zeros((num_molecules, max_molecule_size, len(atom_type2one_hot_index)))
            x_tensor = np.zeros((num_molecules, max_molecule_size, 3))  # For coordinates
            node_mask_tensor = np.zeros((num_molecules, max_molecule_size))

            # Populate tensors
            for i, molecule in enumerate(all_molecules):
                for j, (one_hot, coordinates) in enumerate(molecule):
                    one_hot_tensor[i, j] = one_hot
                    x_tensor[i, j] = coordinates
                    node_mask_tensor[i, j] = 1  # Mark this node as present

            x, one_hot, node_mask = (
                torch.tensor(x_tensor, dtype=dtype, device=device),
                torch.tensor(one_hot_tensor, dtype=dtype, device=device),
                torch.tensor(node_mask_tensor, dtype=dtype, device=device)
            )

            # edge_mask
            bs, n_nodes = node_mask.size()
            edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
            diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
            diag_mask = diag_mask.to(device)
            edge_mask *= diag_mask
            edge_mask = edge_mask.view(bs * n_nodes * n_nodes, 1)

            data = {
                'positions': x.detach(),
                'atom_mask': node_mask.detach(),
                'edge_mask': edge_mask.detach(),
                'one_hot': one_hot.detach(),
            }

            batch_size, n_nodes, _ = data['positions'].size()
            atom_positions = data['positions'].view(batch_size * n_nodes, -1).to(device, torch.float32)
            atom_mask = data['atom_mask'].view(batch_size * n_nodes, -1).to(device, torch.float32)
            edge_mask = data['edge_mask'].to(device, torch.float32)
            nodes = data['one_hot'].to(device, torch.float32)

            nodes = nodes.view(batch_size * n_nodes, -1)
            edges = prop_utils.get_adj_matrix(n_nodes, batch_size, device)

            pred = classifier(
                h0=nodes, 
                x=atom_positions, 
                edges=edges, 
                edge_attr=None, 
                node_mask=atom_mask, 
                edge_mask=edge_mask,
                n_nodes=n_nodes
            )

            return (pred * mad + mean).tolist()

        real_values = get_real_values(image_dir)
        assert len(real_values) == len(image_paths)

    assert len(values) == len(image_paths)

    # Function to apply rounded corners to an image
    def apply_rounded_corners(image, radius):
        # Create mask with rounded corners
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rounded_rectangle([(0, 0), image.size], radius=radius, fill=255)
        
        # Apply rounded corners mask
        image = image.convert("RGBA")
        rounded_image = Image.new("RGBA", image.size)
        rounded_image.paste(image, (0, 0), mask=mask)
        return rounded_image

    # Process all images and store them in a list
    processed_images = []
    for i, image_path in enumerate(image_paths):
        # Load the image and apply rounded corners
        img = Image.open(image_path)
        rounded_img = apply_rounded_corners(img, radius=50)
        
        # Convert the rounded image to a numpy array
        img_np = np.array(rounded_img)
        
        processed_images.append(img_np)

    return processed_images, values, real_values
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

def plot_grid(all_images, all_values, all_real_values, save_path="sweep_grid.pdf"):
    num_rows = len(all_images)
    num_cols = len(all_images[0]) if num_rows > 0 else 0

    # Convert lists to numpy arrays for easier computation
    values_array = np.array(all_values)  # Shape: (num_rows, num_cols)
    real_values_array = np.array(all_real_values)  # Shape: (num_rows, num_cols)

    # Compute per-column means
    mean_values = values_array.mean(axis=0)  # Shape: (num_cols,)
    mean_errors = np.abs(real_values_array - values_array).mean(axis=0)  # Shape: (num_cols,)

    # Create a GridSpec with two extra rows at the top for texts and arrow
    # Height ratios: texts (0.5), arrow (0.2), images (1 each)
    height_ratios = [0.5, 0.2] + [1] * num_rows
    fig_height = 2 + num_rows  # Adjust as needed
    fig_width = 1.2 * num_cols  # Adjust as needed
    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
    gs = gridspec.GridSpec(num_rows + 2, num_cols, height_ratios=height_ratios, hspace=0.3, wspace=0.1)

    # Plot the mean values and mean errors as text above each column in the first row
    for col in range(num_cols):
        ax = fig.add_subplot(gs[0, col])
        ax.set_axis_off()
        # First line: Mean of Values
        ax.text(0.5, 0.2, f"{mean_values[col]:.2f}", 
                ha='center', va='center', fontsize=10, transform=ax.transAxes)
        # Second line: Mean of Errors in green
        # ax.text(0.5, 0.3, f"{mean_errors[col]:.2f}", 
                # ha='center', va='center', fontsize=10, color='#2ca02c', transform=ax.transAxes)

    # Plot the arrow spanning all columns in the second row
    ax_arrow = fig.add_subplot(gs[1, :])
    ax_arrow.set_axis_off()

    # Draw the arrow from left to right across the entire row
    ax_arrow.annotate('', xy=(1.0, 0.5), xytext=(0.0, 0.5),
                      xycoords='axes fraction',
                      arrowprops=dict(facecolor='black', shrink=0.05, headwidth=10, width=1))

    # Plot the images in the grid starting from the third row
    for row in range(num_rows):
        for col in range(num_cols):
            ax = fig.add_subplot(gs[row + 2, col])
            ax.set_axis_off()  # Hide axes

            # Display the image
            img = all_images[row][col]
            ax.imshow(img)

            # Add the real_value below the image (if present)
            if all_real_values is not None:
                ax.text(0.5, -0.09, f"{all_real_values[row][col]:.2f}", 
                        ha='center', va='top', transform=ax.transAxes, fontsize=8, color='#2ca02c')

    # Save the figure with minimal margins
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close(fig)
if __name__ == "__main__":
    image_dirs = [
        # "/home/lizian/Self-Conditioned-3D-Diffusion/eval_src/visualize_results/conditional_sweep_multi/conditional_sweep_2024-09-23_06-25-59",
        "/home/lizian/Self-Conditioned-3D-Diffusion/eval_src/visualize_results/conditional_sweep_multi/conditional_sweep_2024-09-23_06-27-56",
        # "/home/lizian/Self-Conditioned-3D-Diffusion/eval_src/visualize_results/conditional_sweep_multi/conditional_sweep_2024-09-23_06-35-43",
        "/home/lizian/Self-Conditioned-3D-Diffusion/eval_src/visualize_results/conditional_sweep_multi/conditional_sweep_2024-09-23_06-31-50",
        "/home/lizian/Self-Conditioned-3D-Diffusion/eval_src/visualize_results/conditional_sweep_multi/conditional_sweep_2024-09-23_06-29-54",
        # "/home/lizian/Self-Conditioned-3D-Diffusion/eval_src/visualize_results/conditional_sweep_multi/conditional_sweep_2024-09-23_06-37-39",
        "/home/lizian/Self-Conditioned-3D-Diffusion/eval_src/visualize_results/conditional_sweep_multi/conditional_sweep_2024-09-23_06-39-34",
        "/home/lizian/Self-Conditioned-3D-Diffusion/eval_src/visualize_results/conditional_sweep_multi/conditional_sweep_2024-09-23_06-33-47",
        "/home/lizian/Self-Conditioned-3D-Diffusion/eval_src/visualize_results/conditional_sweep_multi/conditional_sweep_2024-09-23_06-41-29",
        "/home/lizian/Self-Conditioned-3D-Diffusion/eval_src/visualize_results/conditional_sweep_multi/conditional_sweep_2024-09-23_06-43-24",
        "/home/lizian/Self-Conditioned-3D-Diffusion/eval_src/visualize_results/conditional_sweep_multi/conditional_sweep_2024-09-23_06-47-14",
        "/home/lizian/Self-Conditioned-3D-Diffusion/eval_src/visualize_results/conditional_sweep_multi/conditional_sweep_2024-09-23_06-45-20",
        "/home/lizian/Self-Conditioned-3D-Diffusion/eval_src/visualize_results/conditional_sweep_multi/conditional_sweep_2024-09-23_06-49-10",
        "/home/lizian/Self-Conditioned-3D-Diffusion/eval_src/visualize_results/conditional_sweep_multi/conditional_sweep_2024-09-23_06-51-04",
        "/home/lizian/Self-Conditioned-3D-Diffusion/eval_src/visualize_results/conditional_sweep_multi/conditional_sweep_2024-09-23_06-52-59",
        # "/home/lizian/Self-Conditioned-3D-Diffusion/eval_src/visualize_results/conditional_sweep_multi/conditional_sweep_2024-09-23_06-54-53",
    ]

    all_images = []
    all_values = []
    all_real_values = []

    for image_dir in image_dirs:
        images, values, real_values = once(image_dir, calculate_real_values, property, class_dir, device)
        all_images.append(images)
        all_values.append(values)
        if calculate_real_values:
            all_real_values.append(real_values)

    # Verify that all directories have the same number of images
    num_cols = len(all_values[0])
    for vals in all_values:
        assert len(vals) == num_cols, "All image directories must have the same number of images."

    if calculate_real_values:
        for real_vals in all_real_values:
            assert len(real_vals) == num_cols, "All real_values lists must have the same number of elements."

    # Plot the grid with the arrow and per-column mean values and errors
    plot_grid(all_images, all_values, all_real_values, save_path="sweep_grid.pdf")

    print("Grid plot saved as sweep_grid.pdf")