from flask import Flask, render_template, request
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load model and class labels
# filepath: c:\Users\prasa\OneDrive\Documents\GitHub\HematoVision-Advanced-Blood-Cell-Classification-Using-Transfer-Learning\app.py\app.py
# ...existing code...

# ...existing code...
labels = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

app = Flask(
    __name__,
    template_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), '../templates'))
)

@app.route('/')
def front():
    return render_template('front.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/static/<path:filename>')
def staticfiles(filename):
    from flask import send_from_directory
    return send_from_directory('static', filename)

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "Image not uploaded", 400

    file = request.files['image']
    if file.filename == '':
        return "No file selected", 400

    # Save uploaded image to static folder
    image_path = os.path.join('static', file.filename)
    print("Saving file to:", image_path)
    file.save(image_path)
    print("File exists after save:", os.path.exists(image_path))
    print("Absolute file path:", os.path.abspath(image_path))

    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    model = load_model('blood_cell_model.h5/blood_cell_model.h5')
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    class_name = labels[class_index]

    # Render result page with prediction and image path
    # Convert backslashes to forward slashes for the URL
    image_url = '/' + image_path.replace('\\', '/')
    print("Image path sent to template:", image_url)
    return render_template('result.html', prediction=class_name, image_path=image_url)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
