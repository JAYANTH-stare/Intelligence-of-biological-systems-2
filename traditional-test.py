import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

# Defining the test dataset path
test_dir = r"C:\\Users\\JAYANTH REDDY\\Dropbox\\My PC (DESKTOP-6ULO7HA)\\Downloads\\archive\\tomato\\val"

# Creating an instance of the ImageDataGenerator for testing
test_data_generator = ImageDataGenerator(rescale=1.0 / 255)

# Loading the test dataset using the ImageDataGenerator
test_dataset = test_data_generator.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Loading the saved model
model = tf.keras.models.load_model('model.h5')

# model is compiled before evaluation
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Evaluating the model on the test dataset
loss, accuracy = model.evaluate(test_dataset)
print("\n✅ Test Results:")
print(f"  - Accuracy: {accuracy * 100:.2f}%")
print(f"  - Loss: {loss:.4f}")

# Predict on test dataset
predictions = model.predict(test_dataset)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_dataset.classes

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

# Compute and plot confusion matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=test_dataset.class_indices.keys(), yticklabels=test_dataset.class_indices.keys())
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
