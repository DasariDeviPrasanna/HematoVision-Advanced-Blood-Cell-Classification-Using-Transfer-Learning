from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tempfile
import base64
from io import BytesIO

# Load model and class labels
labels = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

app = Flask(__name__)

# Configure template and static folders for Vercel
app.template_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates')
app.static_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static')

@app.route('/')
def front():
    return render_template('front.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "Image not uploaded"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            file.save(tmp_file.name)
            tmp_path = tmp_file.name

        try:
            # Load and preprocess the image
            img = image.load_img(tmp_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            # Load model (adjust path for Vercel)
            model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'blood_cell_model.h5', 'blood_cell_model.h5')
            model = load_model(model_path)
            
            # Predict
            prediction = model.predict(img_array)
            class_index = np.argmax(prediction)
            class_name = labels[class_index]
            confidence = float(prediction[0][class_index])

            # Convert image to base64 for display
            img_buffer = BytesIO()
            img.save(img_buffer, format='JPEG')
            img_str = base64.b64encode(img_buffer.getvalue()).decode()

            return render_template('result.html', 
                                prediction=class_name, 
                                image_path=f"data:image/jpeg;base64,{img_str}",
                                confidence=f"{confidence:.2%}")

        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except Exception as e:
        return jsonify({"error": f"Error processing image: {str(e)}"}), 500

# Vercel serverless function handler
def handler(request, context):
    return app(request, context)

if __name__ == '__main__':
    app.run(debug=True)
