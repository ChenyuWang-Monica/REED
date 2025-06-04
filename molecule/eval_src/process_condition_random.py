import sys
sys.path.append(".")
from omegaconf import OmegaConf
from qm9.models import get_model, DistributionProperty
from qm9.property_prediction import prop_utils
from eval_src.eval_conditional_qm9 import get_classifier, get_dataloader
from qm9.utils import compute_mean_mad

import matplotlib.pyplot as plt
import os
import glob
import random
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import torch 
import pickle
# Image directory
image_dir = "eval_src/visualize_results/conditional_random_2024-09-27_08-41-56"
dtype = torch.float32
property = "alpha"
class_dir = "checkpoints/classifiers_ckpts/exp_class_alpha"
device = "cuda"


# Get all image paths
image_paths = sorted(glob.glob(os.path.join(image_dir, "*[0-9][0-9][0-9].png")))

# Number of images to display
column_image_num = 8  # Number of images per row
column_num = 5        # Number of rows

num_total = column_num * column_image_num

# Check if there are enough images
assert len(image_paths) == num_total, "Not enough images to display"


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

# Function to display the image with rounded corners at a specific position
def display_image(ax, image_path, cv, rv, corner_radius=50):
    img = Image.open(image_path)
    
    # Apply rounded corners
    rounded_img = apply_rounded_corners(img, radius=corner_radius)
    
    # Convert PIL image to numpy array for displaying with Matplotlib
    img_np = np.array(rounded_img)
    
    ax.imshow(img_np)
    ax.axis('off')  # Remove axis lines and labels
    
    ax.text(0.5, -0.05, f'{round(rv, 2)}', ha='center', va='top', transform=ax.transAxes, color='black', fontsize=18)
    ax.text(0.5, -0.2, f'{round(cv, 2)}', ha='center', va='top', transform=ax.transAxes, color='green', fontsize=18)



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

with open(f"{image_dir}/prop.pickle", "rb") as f:
    real_values = pickle.load(f) # A list
calculated_values = get_real_values(image_dir)

assert len(real_values) == len(calculated_values) == len(image_paths)

errors = [calculated_values[i] - real_values[i] for i in range(len(real_values))]


# Create a figure with the defined number of subplots
fig, axes = plt.subplots(column_num, column_image_num, figsize=(15, 12))

# Flatten axes array for easier iteration
axes = axes.flatten()

# Loop over the images and axes to display them
for ax, img_path, cv, rv in zip(axes, image_paths, calculated_values, real_values):
    display_image(ax, img_path, cv, rv)

# Adjust spacing between images
plt.tight_layout()

# Show the final image grid
save_path = Path(image_dir) / "conditional_random.pdf"
plt.savefig(save_path, transparent=True,
            bbox_inches='tight',  # Ensure no extra white space around the figure
            pad_inches=0.1,       # Minimal padding
            dpi=500)              # High resolution (300 DPI)