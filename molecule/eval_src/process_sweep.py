import matplotlib.pyplot as plt
import os
import glob
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from pathlib import Path
from PIL import Image, ImageDraw
image_dir = "/home/lizian/Self-Conditioned-3D-Diffusion/eval_src/visualize_results/conditional_sweep_2024-09-27_13-12-41"
calculate_real_values = True
property = "alpha"
class_dir = "checkpoints/classifiers_ckpts/exp_class_alpha"
device = "cuda"

# Get all png images from the directory, assuming filenames are ordered
image_paths = sorted(glob.glob(os.path.join(image_dir, "*[0-9][0-9][0-9].png")))


# Corresponding values to place under each image
values = []
with open(Path(image_dir) / "property_values.log", "r") as f:
    lines = f.readlines()
    for line in lines:
        values.append(float(line))
if calculate_real_values:
    import sys
    sys.path.append(".")
    from omegaconf import OmegaConf
    import numpy as np
    from qm9.models import get_model, DistributionProperty
    import torch
    from qm9.property_prediction import prop_utils

    dtype = torch.float32
    def get_real_values(image_dir):
        xyz_paths = sorted(glob.glob(os.path.join(image_dir, "*[0-9][0-9][0-9].txt")))
        
        from eval_src.eval_conditional_qm9 import get_classifier, get_dataloader    
        from qm9.utils import compute_mean_mad
        classifier = get_classifier(class_dir).to(device)
        # Get generator and dataloader used to train the generator and evalute the classifier
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

        x, one_hot, node_mask = torch.tensor(x_tensor, dtype=dtype, device=device), torch.tensor(one_hot_tensor, dtype=dtype, device=device), torch.tensor(node_mask_tensor, dtype=dtype, device=device)


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
        #charges = data['charges'].to(device, dtype).squeeze(2)
        #nodes = prop_utils.preprocess_input(one_hot, charges, args.charge_power, charge_scale, device)

        nodes = nodes.view(batch_size * n_nodes, -1)
        # nodes = torch.cat([one_hot, charges], dim=1)
        edges = prop_utils.get_adj_matrix(n_nodes, batch_size, device)
        
        pred = classifier(h0=nodes, x=atom_positions, edges=edges, edge_attr=None, node_mask=atom_mask, edge_mask=edge_mask,
                     n_nodes=n_nodes)
        
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

# Create a figure and axis
fig, ax = plt.subplots(figsize=(12, 4))

# Remove the axes (background grid and borders)
ax.set_axis_off()

# Set up image positions
for i, (image_path, value) in enumerate(zip(image_paths, values)):
    # Load the image and apply rounded corners
    img = Image.open(image_path)
    rounded_img = apply_rounded_corners(img, radius=50)
    
    # Convert the rounded image to a numpy array for OffsetImage
    img_np = np.array(rounded_img)
    
    # Add each image with rounded corners
    imagebox = OffsetImage(img_np, zoom=0.15)
    ab = AnnotationBbox(imagebox, (i, 0), frameon=False)
    ax.add_artist(ab)

    # Add corresponding values below each image
    ax.text(i, -0.6, f"{values[i]:.2f}", ha='center', fontsize=12)
    if calculate_real_values:
        # Add corresponding real values further below each image
        ax.text(i, -0.75, f"{real_values[i]:.2f}", ha='center', fontsize=12, color='#2ca02c')  # Add real_values in red for clarity

# Add a dashed line
# ax.plot(range(len(image_paths)), [-0.3]*len(image_paths), 'k--')

# Add an arrow pointing to the right
ax.annotate('', xy=(len(image_paths)-0.4, -0.4), xytext=(-0.6, -0.4),
            arrowprops=dict(facecolor='black', shrink=0.05, headwidth=10, width=1))

# Adjust the limits to fit everything nicely
ax.set_xlim(-1, len(image_paths))
ax.set_ylim(-1, 1)

# Save the final figure
plt.savefig(Path(image_dir) / "sweep.pdf",             
            bbox_inches='tight',  # Ensure no extra white space around the figure
            pad_inches=0,       # Minimal padding
            dpi=800)              # High resolution (300 DPI)