import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load test data
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# Load trained model
model = tf.keras.models.load_model("cat_dog_cnn.keras")

# Class names
class_names = ["Cat", "Dog"]

# Select 10 test images
num_images = 10

predictions = model.predict(X_test[:num_images], verbose=0)

# Convert probabilities to labels
predicted_labels = (predictions >= 0.5).astype(int).flatten()

# Display predictions
plt.figure(figsize=(15, 8))

for i in range(num_images):

    plt.subplot(2, 5, i + 1)

    plt.imshow(X_test[i])

    actual = class_names[y_test[i]]
    predicted = class_names[predicted_labels[i]]

    confidence = predictions[i][0]

    if predicted == "Cat":
        confidence = 1 - confidence

    plt.title(
        f"Actual: {actual}\n"
        f"Predicted: {predicted}\n"
        f"Confidence: {confidence * 100:.1f}%"
    )

    plt.axis("off")

plt.tight_layout()

plt.savefig("sample_predictions.png")

plt.show()

print("Sample predictions saved as: sample_predictions.png")