import numpy as np
import os, requests, cv2, random
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix

# Check if GPUs are available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(f"Using {len(logical_gpus)} GPU(s): {logical_gpus}")
else:
    print("No GPU found. Switching to CPU mode.")

dataset_dir = r"E:\\Desktop\\4th sem\\IBS 2\\project\\archive\\tomato\\train"

data_generator = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

train_dataset = data_generator.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=True,
    subset='training',
    seed=42
)

validation_dataset = data_generator.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=True,
    subset='validation',
    seed=42
)

num_classes = len(train_dataset.class_indices)
print("Number of classes:", num_classes)
class_names = list(train_dataset.class_indices.keys())
print("Class names:", class_names)

# Create a CNN model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

opt = Adam(learning_rate=0.001)
model.compile(
    optimizer=opt,
    loss=tf.keras.losses.categorical_crossentropy,
    metrics=['accuracy']
)
print(model.summary())

# Use `.keras` extension instead of `.h5`
checkpoint_callback = ModelCheckpoint('model.keras', monitor='val_loss', save_best_only=True)

early_stopping_callback = EarlyStopping(
    monitor='val_loss',  
    patience=10,          
    verbose=1,            
    restore_best_weights=True  
)

csv_logger = CSVLogger('training_log.csv')

history = model.fit(
    train_dataset,
    batch_size=32,
    epochs=100, 
    validation_data=validation_dataset, 
    callbacks=[checkpoint_callback, csv_logger, early_stopping_callback]
)

# Plot accuracy vs. epochs
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Epochs')
plt.legend()
plt.savefig('accuracy_vs_epochs.png')  

# Plot loss vs. epochs
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs. Epochs')
plt.legend()
plt.savefig('loss_vs_epochs.png')  
plt.show()
