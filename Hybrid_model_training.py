import sys
import io
import os
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, NASNetMobile
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
import numpy as np

# Force UTF-8 Encoding for stdout and stderr
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Suppress TensorFlow Logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Set Seed for Reproducibility
tf.random.set_seed(42)

# Dataset Paths
dataset_path = r"E:\\Desktop\\4th sem\\IBS 2\\project\\archive (1)\\plantvillage dataset\\color"
filtered_dataset_path = r"E:\\Desktop\\4th sem\\IBS 2\\project\\tomtom"

# Class Names
tomato_classes = [
    'Tomato___healthy',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Target_Spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Leaf_Mold',
    'Tomato___Late_blight',
    'Tomato___Early_blight',
    'Tomato___Bacterial_spot'
]

# Ensure Directory Structure
os.makedirs(filtered_dataset_path, exist_ok=True)

# Copy Relevant Images
print("Copying relevant images...")
for class_name in tomato_classes:
    source_dir = os.path.join(dataset_path, class_name)
    target_dir = os.path.join(filtered_dataset_path, class_name)
    os.makedirs(target_dir, exist_ok=True)
    for file_name in os.listdir(source_dir):
        source_file = os.path.join(source_dir, file_name)
        target_file = os.path.join(target_dir, file_name)
        if os.path.isfile(source_file):
            shutil.copy(source_file, target_file)
print("Image copying completed.")

# Image Data Preprocessing
image_size = (224, 224)
batch_size = 32
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.25,
    height_shift_range=0.25,
    shear_range=0.25,
    zoom_range=0.25,
    horizontal_flip=True,
    vertical_flip=True,  # Vertical flipping added
    brightness_range=[0.8, 1.2],  # Brightness adjustments added
    validation_split=0.2
)

print("Loading training data...")
train_generator = datagen.flow_from_directory(
    filtered_dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)
print("Training data loaded.")

print("Loading validation data...")
validation_generator = datagen.flow_from_directory(
    filtered_dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)
print("Validation data loaded.")

# Compute Class Weights
print("Calculating class weights...")
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))
# Define the Model
# Define the Model
print("Defining the model...")

# Shared input layer
# Define separate input layers for the two inputs
input1 = tf.keras.Input(shape=(224, 224, 3))
input2 = tf.keras.Input(shape=(224, 224, 3))

# VGG16 branch
vgg16_base = VGG16(weights='imagenet', include_top=False, input_tensor=input1)
for layer in vgg16_base.layers[:-8]:
    layer.trainable = False
vgg16_output = GlobalAveragePooling2D()(vgg16_base.output)

# NASNetMobile branch
nasnet_base = NASNetMobile(weights='imagenet', include_top=False, input_tensor=input2)
for layer in nasnet_base.layers[:-8]:
    layer.trainable = False
nasnet_output = GlobalAveragePooling2D()(nasnet_base.output)

# Combine outputs from both models
combined = concatenate([vgg16_output, nasnet_output])

# Fully connected layers
x = Dense(256, activation='relu')(combined)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
output = Dense(train_generator.num_classes, activation='softmax')(x)

# Define the final model
model = Model(inputs=[input1, input2], outputs=output)

# Optimizer with Learning Rate Decay
lr_schedule = ExponentialDecay(
    initial_learning_rate=0.0001,
    decay_steps=10000,
    decay_rate=0.9
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

# Custom Data Generator for Single Input with Sample Weights
def generate_double_input(generator):
    def gen():
        for images, labels in generator:
            yield ((images, images), labels)  # Correct structure: ((input1, input2), labels)
    return tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            (
                tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),  # input1
                tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),  # input2
            ),
            tf.TensorSpec(shape=(None, train_generator.num_classes), dtype=tf.float32),  # labels
        )
    ).prefetch(tf.data.AUTOTUNE)

# Create datasets
train_dataset = generate_double_input(train_generator)
validation_dataset = generate_double_input(validation_generator)

# 6. **Callbacks**
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
tensorboard = TensorBoard(log_dir='logs', write_graph=True)

# Calculate steps per epoch
steps_per_epoch = train_generator.samples // batch_size
validation_steps = validation_generator.samples // batch_size

# Training block with defined steps
print("Starting training...")
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=15,
    callbacks=[early_stopping, reduce_lr, tensorboard],
    verbose=1  # Ensure progress bar is shown
)

print("Training completed.")
model.save("trained_model.h5")