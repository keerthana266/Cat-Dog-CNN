import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt


# ============================================================
# 1. LOAD PROCESSED DATASETS
# ============================================================

print("Loading processed datasets...")

X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")

X_val = np.load("X_val.npy")
y_val = np.load("y_val.npy")

X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

print("Training data:", X_train.shape)
print("Validation data:", X_val.shape)
print("Testing data:", X_test.shape)


# ============================================================
# 2. BUILD CNN MODEL
# ============================================================

model = models.Sequential([

    # First Convolutional Block
    layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(128, 128, 3)
    ),
    layers.MaxPooling2D((2, 2)),

    # Second Convolutional Block
    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),
    layers.MaxPooling2D((2, 2)),

    # Third Convolutional Block
    layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),
    layers.MaxPooling2D((2, 2)),

    # Flatten feature maps
    layers.Flatten(),

    # Fully Connected Layer
    layers.Dense(
        128,
        activation="relu"
    ),

    # Dropout to reduce overfitting
    layers.Dropout(0.5),

    # Output layer
    # 0 = Cat
    # 1 = Dog
    layers.Dense(
        1,
        activation="sigmoid"
    )
])


# ============================================================
# 3. DISPLAY MODEL ARCHITECTURE
# ============================================================

print("\nCNN Model Architecture:\n")

model.summary()


# ============================================================
# 4. COMPILE MODEL
# ============================================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\nCNN model compiled successfully!")


# ============================================================
# 5. TRAIN CNN
# ============================================================

print("\nStarting CNN training...\n")

history = model.fit(
    X_train,
    y_train,
    epochs=15,
    batch_size=32,
    validation_data=(X_val, y_val)
)


# ============================================================
# 6. SAVE TRAINED MODEL
# ============================================================

model.save("cat_dog_cnn.keras")

print("\nModel training completed!")
print("Trained model saved as: cat_dog_cnn.keras")


# ============================================================
# 7. PLOT TRAINING AND VALIDATION ACCURACY
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("CNN Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.savefig("accuracy_plot.png")

plt.show()


# ============================================================
# 8. PLOT TRAINING AND VALIDATION LOSS
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("CNN Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.savefig("loss_plot.png")

plt.show()


# ============================================================
# 9. FINAL VALIDATION EVALUATION
# ============================================================

val_loss, val_accuracy = model.evaluate(
    X_val,
    y_val,
    verbose=1
)

print("\nValidation Results:")
print("Validation Loss:", val_loss)
print("Validation Accuracy:", val_accuracy)