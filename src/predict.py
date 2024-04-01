import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2

# Assuming you know the image dimensions and number of classes from your dataset
image_height = 600  # Adjust based on your data
image_width = 600  # Adjust based on your data
num_classes = 2  # Adjust based on your data (e.g., 2 for binary)

# Load the trained model
model = load_model('best_model.keras')

def predict_image(image_path):
  """
  Preprocesses an image and predicts if it's fake or real.

  Args:
      image_path: Path to the image file.

  Returns:
      A string indicating the predicted class (fake or real).
  """
  # Load the image
  image = cv2.imread(image_path)

  # Preprocess the image (resize, normalization, etc.)
  # ... your preprocessing code based on your training pipeline ...

  # Prepare the input
  image_input = image.reshape([1, image.shape[0], image.shape[1], image.shape[2]])

  # Make a prediction
  prediction = model.predict(image_input)

  # Interpret the prediction (adjust threshold as needed)
  if prediction[0][0] > 0.5:
    return "fake"
  else:
    return "real"

# Example usage (replace with your image path)
image_path = r"C:\Users\abhin\OneDrive\Desktop\deepfake_detection\data\validation\real\mid_179_1111.jpg"


prediction = predict_image(image_path)
print(f"Image predicted to be: {prediction}")
