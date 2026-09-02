import os
import matplotlib.pyplot as plt
from PIL import Image

# Dataset paths
cat_path = "PetImages/Cat"
dog_path = "PetImages/Dog"

# Get image filenames
cat_images = os.listdir(cat_path)
dog_images = os.listdir(dog_path)

# Display 5 cat and 5 dog images
plt.figure(figsize=(12, 6))

# Cat images
for i in range(5):
    image_path = os.path.join(cat_path, cat_images[i])
    image = Image.open(image_path)

    plt.subplot(2, 5, i + 1)
    plt.imshow(image)
    plt.title("Cat")
    plt.axis("off")

# Dog images
for i in range(5):
    image_path = os.path.join(dog_path, dog_images[i])
    image = Image.open(image_path)

    plt.subplot(2, 5, i + 6)
    plt.imshow(image)
    plt.title("Dog")
    plt.axis("off")

plt.tight_layout()
plt.show()