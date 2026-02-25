from flask import Flask, render_template, request, redirect, url_for, session
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from functools import wraps

# Load model and class labels
labels = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

app = Flask(
    __name__,
    template_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), '../templates')),
    static_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), '../static'))
)

# Set secret key for session management
app.secret_key = 'your-secret-key-change-this-in-production-2024'

# Simple user database (in production, use a real database)
USERS = {
    'admin': 'admin123',
    'user': 'user123',
    'demo': 'demo123'
}

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def front():
    return render_template('front.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Check if username exists and password matches
        if username in USERS and USERS[username] == password:
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('home'))
        else:
            return redirect(url_for('login', error=1))
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('front'))

@app.route('/home')
@login_required
def home():
    return render_template('home.html', username=session.get('username'))

# Prediction route
@app.route('/predict', methods=['POST'])
@login_required
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
        return render_template('result.html', prediction=class_name, image_path=image_url, username=session.get('username'))
        
    except Exception as e:
        print(f"Error in predict route: {str(e)}")
        return f"Error processing image: {str(e)}", 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
