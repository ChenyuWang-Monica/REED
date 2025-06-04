import matplotlib.pyplot as plt
import os
import glob
import random
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

# Image directory
image_dir = "./eval_src/visualize_results/unconditional_random_2024-09-29_06-43-59"

# Get all image paths
image_paths = sorted(glob.glob(os.path.join(image_dir, "*[0-9][0-9][0-9].png")))

# Number of images to display
column_image_num = 8  # Number of images per row
column_num = 5        # Number of rows

num_total = column_num * column_image_num

# Check if there are enough images
assert len(image_paths) >= num_total, "Not enough images to display"

# Randomly select `num_total` images
# selected_images = random.sample(image_paths, num_total)
selected_images = image_paths
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
def display_image(ax, image_path, corner_radius=50):
    img = Image.open(image_path)
    
    # Apply rounded corners
    rounded_img = apply_rounded_corners(img, radius=corner_radius)
    
    # Convert PIL image to numpy array for displaying with Matplotlib
    img_np = np.array(rounded_img)
    
    ax.imshow(img_np)
    ax.axis('off')  # Remove axis lines and labels

# Create a figure with the defined number of subplots
fig, axes = plt.subplots(column_num, column_image_num, figsize=(15, 12))

# Flatten axes array for easier iteration
axes = axes.flatten()

# Loop over the images and axes to display them
for ax, img_path in zip(axes, selected_images):
    display_image(ax, img_path)

# Adjust spacing between images
plt.tight_layout()

# Show the final image grid
save_path = Path(image_dir) / "unconditional_random.pdf"
plt.savefig(save_path, transparent=True,
            bbox_inches='tight',  # Ensure no extra white space around the figure
            pad_inches=0.1,       # Minimal padding
            dpi=500)              # High resolution (300 DPI)