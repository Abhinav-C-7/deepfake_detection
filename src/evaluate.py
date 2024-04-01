import tensorflow as tf
from model import create_model  # Import your model architecture
from data_loader import test_generator  # Import test data generator

# Assuming you know the image dimensions and number of classes from your dataset
image_height = 600  # Adjust based on your data
image_width = 600  # Adjust based on your data
num_classes = 2  # Adjust based on your data (e.g., 2 for binary)

# Load the best model saved during training
model = create_model(image_height, image_width, num_classes)
model.load_weights('best_model.keras')

# Compile the model (same loss and metrics as training)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_generator)

print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")
