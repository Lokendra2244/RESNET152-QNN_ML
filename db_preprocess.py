# %% Cell 1: library imports for database prcessing
import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# %% Cell 2: script flexibility for local testing or HPC testing
TEST_MODE = True
while True:
    uc = input(
        "do you want to run this in test mode with 20 images?(y/n) :"
    )  # user choice
    if uc == "y":
        TEST_MODE = True
        break
    elif uc == "n":
        TEST_MODE = False
        break
    print("invalid input please try again")
# %% Cell 3: data loading and filtering
db = pd.read_csv("./archive/csv/mass_case_description_train_set.csv")
db = db[["pathology", "cropped image file path"]]
if TEST_MODE:
    db = db.head(20)
else:
    pass

# %% Cell 4: cleaning the table
mapping = {"BENIGN": 1, "BENIGN_WITHOUT_CALLBACK": 1, "MALIGNANT": 0}
db["pathology"] = db["pathology"].map(mapping)

# %% Cell 5: function for finding  jpeg files and fixing dicom paths in database


def fix_path(path):
    parts = str(path).split("/")
    for part in parts:
        if part.startswith("1.3.6"):
            possible_path = os.path.join("./archive/jpeg/", part)
            if os.path.exists(possible_path):
                images = os.listdir(possible_path)
                lst_file = None  # complete path of the correct file in the possible_path directory
                min_pxl = float("inf")
                for img in images:
                    full_img_path = os.path.join(possible_path, img)
                    try:
                        # Open image and calculate width x height
                        with Image.open(full_img_path) as opened_img:
                            width, height = opened_img.size
                            total_pixels = width * height
                        # If this image has fewer pixels, it's our new best guess for the crop
                        if total_pixels < min_pxl:
                            min_pxl = total_pixels
                            lst_file = full_img_path
                    except Exception as e:
                        # Ignore any hidden non-image files
                        continue
                if lst_file is not None:
                    return lst_file
    return None


db["cropped image file path"] = db["cropped image file path"].apply(fix_path)
db = db.dropna()

# %% Cell 6 : saving the data frame to a CSV file
if TEST_MODE:
    db.to_csv("clean_data_set.csv", index=False)
else:
    db.to_csv("clean_data_set_full.csv", index=False)

# %% cell 7: printing image grid to confirm the data is correct

# Load the CSV that was just generated
csv_to_check = "clean_data_set.csv" if TEST_MODE else "clean_data_set_full.csv"
verify_db = pd.read_csv(csv_to_check)

# Grab the first 16 images to make a 4x4 grid
sample_paths = verify_db.head(16)["cropped image file path"].values
sample_labels = verify_db.head(16)["pathology"].values

fig, axes = plt.subplots(4, 4, figsize=(10, 10))
fig.suptitle("Data Pipeline Verification: Cropped Tissues", fontsize=16)

for i, ax in enumerate(axes.flat):
    if i < len(sample_paths):
        # Open the physical image file
        img = Image.open(sample_paths[i])
        ax.imshow(img, cmap="gray")

        # Add the label (1 = Benign, 0 = Malignant)
        label_text = "Benign" if sample_labels[i] == 1 else "Malignant"
        ax.set_title(label_text)
    ax.axis("off")  # Hide gridlines and axes

plt.tight_layout()
plt.show()
