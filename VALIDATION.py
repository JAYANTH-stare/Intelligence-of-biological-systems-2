import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load the trained model
model_path = "E:\\Desktop\\4th sem\\IBS 2\\project\\trained_model.h5"
model = tf.keras.models.load_model(model_path)

# Define the validation dataset path
validation_dir = "E:\\Desktop\\4th sem\\IBS 2\\project\\archive\\tomato\\val" 

# Image data generator for validation 
validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Load validation dataset
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(256, 256), 
    batch_size=32,
    class_mode='categorical', 
    shuffle=False
)

# model is compiled before evaluation
model.compile(loss="categorical_crossentropy", metrics=["accuracy"])

# Evaluate the model
loss, accuracy = model.evaluate(validation_generator)
print("\n✅ Validation Results:")
print(f"  - Accuracy: {accuracy*100:.2f}%")
print(f"  - Loss: {loss:.4f}")

# Predict on validation dataset
predictions = model.predict(validation_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = validation_generator.classes

# Compute Precision, Recall, and F1-score
precision = precision_score(true_classes, predicted_classes, average='weighted')
recall = recall_score(true_classes, predicted_classes, average='weighted')
f1 = f1_score(true_classes, predicted_classes, average='weighted')

print("\n🔍 Performance Metrics:")
print(f"  - Precision: {precision:.4f}")
print(f"  - Recall: {recall:.4f}")
print(f"  - F1 Score: {f1:.4f}")

# Print detailed classification report
print("\n🔍 Classification Report:")
print(classification_report(true_classes, predicted_classes))

# Display confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=validation_generator.class_indices.keys(), yticklabels=validation_generator.class_indices.keys())
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
