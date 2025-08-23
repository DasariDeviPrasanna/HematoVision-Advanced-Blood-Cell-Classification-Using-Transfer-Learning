from flask import Flask, render_template, request
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load model and class labels
labels = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

app = Flask(
    __name__,
    template_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), '../templates')),
    static_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), '../static'))
)

@app.route('/')
def front():
    return render_template('front.html')

@app.route('/home')
def home():
    return render_template('home.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return "Image not uploaded", 400

        file = request.files['image']
        if file.filename == '':
            return "No file selected", 400

        # Create static directory if it doesn't exist
        static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../static'))
        os.makedirs(static_dir, exist_ok=True)

        # Save uploaded image to static folder with proper path handling
        image_path = os.path.join(static_dir, file.filename)
        print("Saving file to:", image_path)
        file.save(image_path)
        print("File exists after save:", os.path.exists(image_path))
        print("Absolute file path:", image_path)

        # Load and preprocess the image
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Load model with proper path handling
        model_path = os.path.join(os.path.dirname(__file__), '..', 'blood_cell_model.h5', 'blood_cell_model.h5')
        print("Loading model from:", model_path)
        
        if not os.path.exists(model_path):
            return "Model file not found. Please ensure blood_cell_model.h5 exists.", 500
            
        model = load_model(model_path)
        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)
        class_name = labels[class_index]

        # Render result page with prediction and image path
        # Use relative path for the template
        image_url = '/static/' + file.filename
        print("Image URL sent to template:", image_url)
        return render_template('result.html', prediction=class_name, image_path=image_url)
        
    except Exception as e:
        print(f"Error in predict route: {str(e)}")
        return f"Error processing image: {str(e)}", 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
