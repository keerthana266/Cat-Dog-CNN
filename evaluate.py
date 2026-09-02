import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt


# ============================================================
# 1. LOAD TEST DATA
# ============================================================

print("Loading test dataset...")

X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

print("Test data shape:", X_test.shape)
print("Test labels shape:", y_test.shape)


# ============================================================
# 2. LOAD TRAINED CNN MODEL
# ============================================================

print("\nLoading trained CNN model...")

model = tf.keras.models.load_model("cat_dog_cnn.keras")

print("Model loaded successfully!")


# ============================================================
# 3. MAKE PREDICTIONS
# ============================================================

print("\nMaking predictions on test data...")

# Model gives probability between 0 and 1
y_probability = model.predict(X_test)

# Convert probabilities into class labels
# < 0.5 = Cat (0)
# >= 0.5 = Dog (1)

y_pred = (y_probability >= 0.5).astype(int).flatten()

print("Predictions completed!")


# ============================================================
# 4. CALCULATE ACCURACY
# ============================================================

accuracy = accuracy_score(y_test, y_pred)


# ============================================================
# 5. CALCULATE PRECISION
# ============================================================

precision = precision_score(y_test, y_pred)


# ============================================================
# 6. CALCULATE RECALL
# ============================================================

recall = recall_score(y_test, y_pred)


# ============================================================
# 7. CALCULATE F1-SCORE
# ============================================================

f1 = f1_score(y_test, y_pred)


# ============================================================
# 8. DISPLAY RESULTS
# ============================================================

print("\n========================================")
print("       CNN MODEL TEST RESULTS")
print("========================================")

print(f"Accuracy  : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision : {precision:.4f} ({precision * 100:.2f}%)")
print(f"Recall    : {recall:.4f} ({recall * 100:.2f}%)")
print(f"F1-Score  : {f1:.4f} ({f1 * 100:.2f}%)")

print("========================================")


# ============================================================
# 9. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Cat", "Dog"]
    )
)


# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# 11. DISPLAY CONFUSION MATRIX
# ============================================================

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("Confusion Matrix - Cat and Dog Classification")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.xticks(
    [0, 1],
    ["Cat", "Dog"]
)

plt.yticks(
    [0, 1],
    ["Cat", "Dog"]
)

# Display numbers inside matrix
for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()

plt.tight_layout()

plt.savefig("confusion_matrix.png")

plt.show()


# ============================================================
# 12. SAVE RESULTS TO TEXT FILE
# ============================================================

with open("results.txt", "w") as file:

    file.write("CAT AND DOG CLASSIFICATION USING CNN\n")
    file.write("====================================\n\n")

    file.write(f"Test Images: {len(X_test)}\n\n")

    file.write(
        f"Accuracy  : {accuracy:.4f} ({accuracy * 100:.2f}%)\n"
    )

    file.write(
        f"Precision : {precision:.4f} ({precision * 100:.2f}%)\n"
    )

    file.write(
        f"Recall    : {recall:.4f} ({recall * 100:.2f}%)\n"
    )

    file.write(
        f"F1-Score  : {f1:.4f} ({f1 * 100:.2f}%)\n"
    )

print("\nResults saved to: results.txt")
print("Confusion matrix saved to: confusion_matrix.png")