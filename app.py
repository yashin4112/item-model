# from flask import Flask, render_template, request, jsonify
# import tensorflow as tf
# from werkzeug.utils import secure_filename
# import numpy as np
# import os

# app = Flask(__name__)

# UPLOAD_FOLDER = "./files"
# # Load the fine-tuned model
# model = tf.keras.models.load_model('./fine_tuned_mobilenet_grocery.h5')

# # Load class labels (update this to match your dataset structure)
# class_labels = ['buds_1750', 'watch_2000']  # Replace with your actual class names

# def predict_grocery_object(image_path):
#     """
#     Predict the grocery object class from the image.
#     """
#     # Load and preprocess the image
#     img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
#     img_array = tf.keras.preprocessing.image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = tf.keras.applications.mobilenet.preprocess_input(img_array)

#     # Predict the class
#     predictions = model.predict(img_array)
#     predicted_class = np.argmax(predictions[0])

#     return class_labels[predicted_class]


# @app.route('/')
# def home():
#     return render_template('predict.html')


# @app.route('/predict', methods=['POST'])
# def predict():

#     if 'image' not in request.files:
#         return jsonify({"error": "No image file provided"}), 400



#     # Get the uploaded file
#     img = request.files['image']
#     img_filename = secure_filename(img.filename)
#     img_path = os.path.join(UPLOAD_FOLDER, img_filename)
#     img.save(img_path)

#     # Save the file to a temporary location
#     try:
#         # Predict the object class
#         predicted_label = predict_grocery_object(UPLOAD_FOLDER + '/' + img_filename)

#         # Return the result
#         return jsonify({"object": predicted_label}), 200

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

#     # finally:
#         # Clean up the temporary file
#         # if os.path.exists(UPLOAD_FOLDER):
#         #     os.remove(UPLOAD_FOLDER + '/' + img_filename)

# if __name__ == '__main__':
#     app.run(host= "0.0.0.0",debug=False,port=5001)


from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from werkzeug.utils import secure_filename
import numpy as np
import os

app = Flask(__name__)

UPLOAD_FOLDER = "./files"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the fine-tuned model
model = tf.keras.models.load_model('./old_fine_tuned_mobilenet_grocery.h5')

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
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the request contains the 'image' key in files
    if 'image' not in request.files:
        print(request.files, "No image files provided")
        return jsonify({"error": "No image files provided"}), 400

    # Get the list of uploaded files
    uploaded_files = request.files.getlist('image')
    if not uploaded_files:
        print(request.files, "No files found in request")
        return jsonify({"error": "No files found in request"}), 400

    predictions = []
    for img in uploaded_files:
        try:
            # Save each uploaded image
            img_filename = secure_filename(img.filename)
            img_path = os.path.join(UPLOAD_FOLDER, img_filename)
            img.save(img_path)

            # Predict the object class
            predicted_label = predict_grocery_object(img_path)

            # Append the prediction label to the list
            predictions.append(predicted_label)

            # Clean up the processed image
            # os.remove(img_path)
        except Exception as e:
            # If there's an error, append a placeholder or skip the file
            predictions.append(f"Error processing {img.filename}: {str(e)}")

    # Return only the list of prediction labels
    print(predictions)
    return jsonify(predictions), 200


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=5001)
