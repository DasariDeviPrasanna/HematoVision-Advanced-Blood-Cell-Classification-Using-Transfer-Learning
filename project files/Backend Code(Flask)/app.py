from flask import Flask, request, render_template, redirect, url_for
import os
import numpy as np
import cv2
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__, template_folder='../Frontend(templates)', static_folder='static')
model = load_model("../Model Training/BloodCell.h5")

# Create a 'static/uploads' folder if it doesn't exist
uploads_folder = os.path.join('static', 'uploads')
if not os.path.exists(uploads_folder):
    os.makedirs(uploads_folder)

class_labels = ['eosinophil', 'lymphocyte', 'monocyte', 'neutrophil']

def predict_image_class(image_path, model):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_preprocessed = preprocess_input(img_resized.reshape((1, 224, 224, 3)))
    predictions = model.predict(img_preprocessed)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    predicted_class_label = class_labels[predicted_class_idx]
    return predicted_class_label

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(uploads_folder, filename)
            file.save(file_path)
            predicted_class_label = predict_image_class(file_path, model)
            
            image_file_path = os.path.join('uploads', filename)

            return render_template("result.html", class_label=predicted_class_label, image_file=image_file_path)
    return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True)
