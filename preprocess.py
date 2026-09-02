import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# Dataset paths
cat_path = "PetImages/Cat"
dog_path = "PetImages/Dog"

# Image size
IMG_SIZE = 128

images = []
labels = []

# -------------------------
# Load Cat Images
# -------------------------
print("Loading cat images...")

for filename in os.listdir(cat_path):
    file_path = os.path.join(cat_path, filename)

    try:
        image = Image.open(file_path).convert("RGB")
        image = image.resize((IMG_SIZE, IMG_SIZE))

        images.append(np.array(image))
        labels.append(0)

    except Exception:
        print("Skipping invalid image:", filename)


# -------------------------
# Load Dog Images
# -------------------------
print("Loading dog images...")

for filename in os.listdir(dog_path):
    file_path = os.path.join(dog_path, filename)

    try:
        image = Image.open(file_path).convert("RGB")
        image = image.resize((IMG_SIZE, IMG_SIZE))

        images.append(np.array(image))
        labels.append(1)

    except Exception:
        print("Skipping invalid image:", filename)


# Convert to NumPy arrays
X = np.array(images, dtype=np.float32)
y = np.array(labels)

# Normalize pixel values
X = X / 255.0

print("\nDataset loaded successfully!")
print("Images shape:", X.shape)
print("Labels shape:", y.shape)

# -------------------------
# Split into Train + Temp
# -------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# -------------------------
# Split Temp into Validation + Test
# -------------------------
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nDataset Split:")
print("Training images:", len(X_train))
print("Validation images:", len(X_val))
print("Testing images:", len(X_test))

print("\nImage shape:", X_train.shape[1:])
# Save processed datasets
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)

np.save("X_val.npy", X_val)
np.save("y_val.npy", y_val)

np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

print("\nProcessed datasets saved successfully!")