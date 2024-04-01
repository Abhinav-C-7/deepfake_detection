import tensorflow as tf
from tensorflow.keras import layers

def create_model(image_height, image_width, num_classes):
    inputs = layers.Input(shape=(600, 600, 3))  # Adjust for image format

    # Example convolutional blocks (modify as needed)
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Example dense layers (modify as needed)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)

    # Output layer (adjust for binary/multi-class)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
