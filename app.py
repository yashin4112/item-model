from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from werkzeug.utils import secure_filename
import numpy as np
import os

app = Flask(__name__)

UPLOAD_FOLDER = "./files"
# Load the fine-tuned model
model = tf.keras.models.load_model('fine_tuned_mobilenet_grocery.h5')

# Load class labels (update this to match your dataset structure)
class_labels = ['buds_1750', 'watch_2000']  # Replace with your actual class names

def predict_grocery_object(image_path):
    """
    Predict the grocery object class from the image.
    """
    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet.preprocess_input(img_array)

    # Predict the class
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])

    return class_labels[predicted_class]


@app.route('/')
def home():
    print("get request coming............")
    return render_template('predict.html')


@app.route('/predict', methods=['POST'])
def predict():
    print("request coming............")
    
    """
    API endpoint to predict the grocery object from an uploaded image.
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400



    # Get the uploaded file
    img = request.files['image']
    img_filename = secure_filename(img.filename)
    img_path = os.path.join(UPLOAD_FOLDER, img_filename)
    img.save(img_path)

    # Save the file to a temporary location
    try:
        # Predict the object class
        predicted_label = predict_grocery_object(UPLOAD_FOLDER + '/' + img_filename)

        # Return the result
        return jsonify({"object": predicted_label}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # finally:
        # Clean up the temporary file
        # if os.path.exists(UPLOAD_FOLDER):
        #     os.remove(UPLOAD_FOLDER + '/' + img_filename)

if __name__ == '__main__':
    app.run(host= "0.0.0.0",debug=False,port=3000)
