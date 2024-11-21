# predict.py

import tensorflow as tf
import numpy as np
import os

# Load the fine-tuned model
model = tf.keras.models.load_model('fine_tuned_mobilenet_grocery.h5')

def predict_grocery_object(image_path):
    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet.preprocess_input(img_array)

    # Predict the class
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    
    # Load class labels (this depends on the dataset directory structure)
    class_labels = ['buds', 'watch']  # Replace with actual class names

    return class_labels[predicted_class]

# Test with a new image
# test_image_path = 'test_images/oppobuds.png'  # Update with your test image path
# predicted_label = predict_grocery_object(test_image_path)
# print(f"The object is recognized as: {predicted_label}")
