import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import math

# Load the full dataset
db = pd.read_csv("clean_data_set_full.csv")
paths = db["cropped image file path"].values
labels = db["pathology"].values

# Setup grid dimensions
rows, cols = 6, 6
grid_size = rows * cols
total_pages = math.ceil(len(paths) / grid_size)

print(f"Loaded {len(paths)} images. Launching viewer...")

# Loop through the dataset in chunks of 36
for page in range(total_pages):
    start_idx = page * grid_size
    end_idx = min(start_idx + grid_size, len(paths))

    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    fig.suptitle(
        f"Full Dataset Explorer: Page {page + 1} of {total_pages}\n(Close window to see next page)",
        fontsize=16,
    )

    for i, ax in enumerate(axes.flat):
        if start_idx + i < end_idx:
            # Open and plot the image
            img = Image.open(paths[start_idx + i])
            ax.imshow(img, cmap="gray")

            # Add the label
            lbl = "Benign" if labels[start_idx + i] == 1 else "Malignant"
            ax.set_title(lbl, fontsize=10)

        ax.axis("off")  # Hide gridlines

    plt.tight_layout()
    plt.show()  # Script completely pauses here until you close the UI window
