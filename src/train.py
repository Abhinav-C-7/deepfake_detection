import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from model import create_model  # Import your model architecture
from data_loader import train_generator, validation_generator  # Import data generators

# Hyperparameters (adjust based on your dataset and hardware)
epochs = 3
batch_size = 16
learning_rate = 0.001
images, targets = next(train_generator)
print(f"Image shape: {images.shape}")
print(f"Target shape: {targets.shape}")
# Define optimizer (Adam is a common choice)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Assuming you know the image dimensions and number of classes from your dataset
image_height = 600  # Adjust based on your data
image_width = 600  # Adjust based on your data
num_classes = 2      # Adjust based on your data (e.g., 2 for binary)

# Create the model instance with required arguments
model = create_model(image_height, image_width, num_classes)

# Compile the model (categorical crossentropy for binary classification)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# ... rest of your training script continues as before ...

# Early stopping callback to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Model checkpoint callback to save the best model based on validation loss
model_checkpoint = ModelCheckpoint(filepath='best_model.keras',  # Fixed filepath extension
                                   monitor='val_loss',
                                   save_best_only=True)

# Train the model
history = model.fit(train_generator,
                    epochs=epochs,
                    validation_data=validation_generator,
                    callbacks=[early_stopping, model_checkpoint])


# Optional: Print training history and save final model (code remains the same)

