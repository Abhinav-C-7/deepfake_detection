import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data directories (adjust based on your project structure)
DATA_DIR = "data"  # Replace with the actual path
IMAGE_SIZE = (600, 600)  # Target image size

# Data generator with augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Data generator for validation and testing (no augmentation)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load train, validation, and test images (including class labels)
train_generator = train_datagen.flow_from_directory(
    DATA_DIR + "/train",
    target_size=IMAGE_SIZE,
    batch_size=16,
    class_mode='binary',  # Assuming binary classification
    classes=['real', 'fake']  # Replace with your class names
)

validation_generator = test_datagen.flow_from_directory(
    DATA_DIR + "/validation",
    target_size=IMAGE_SIZE,
    batch_size=16,
    class_mode='binary',  # Assuming binary classification
    classes=['real', 'fake']  # Replace with your class names
)

test_generator = test_datagen.flow_from_directory(
    DATA_DIR + "/test",
    target_size=IMAGE_SIZE,
    batch_size=16,
    class_mode='binary',  # Assuming binary classification
    classes=['real', 'fake']  # Replace with your class names
)
