from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from functools import wraps
import tempfile
import base64
from io import BytesIO

# Load model and class labels
labels = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

# Get the project root directory (parent of api folder)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(
    __name__,
    template_folder=os.path.join(PROJECT_ROOT, 'templates'),
    static_folder=os.path.join(PROJECT_ROOT, 'static')
)

# Set secret key for session management
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-this-in-production-2024')

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

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "Image not uploaded"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Use temporary file (Vercel serverless has read-only filesystem except /tmp)
        tmp_dir = '/tmp' if os.path.exists('/tmp') else tempfile.gettempdir()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg', dir=tmp_dir) as tmp_file:
            file.save(tmp_file.name)
            tmp_path = tmp_file.name

        try:
            # Load and preprocess the image
            img = image.load_img(tmp_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            # Load model - Vercel includes all project files
            model_path = os.path.join(PROJECT_ROOT, 'blood_cell_model.h5', 'blood_cell_model.h5')
            if not os.path.exists(model_path):
                return jsonify({"error": "Model file not found"}), 500

            model = load_model(model_path)
            prediction = model.predict(img_array)
            class_index = np.argmax(prediction)
            class_name = labels[class_index]
            confidence = float(prediction[0][class_index])

            # Convert image to base64 for display (avoids static file storage in serverless)
            img_obj = image.load_img(tmp_path, target_size=(224, 224))
            img_buffer = BytesIO()
            img_obj.save(img_buffer, format='JPEG')
            img_str = base64.b64encode(img_buffer.getvalue()).decode()
            image_data = f"data:image/jpeg;base64,{img_str}"

            return render_template('result.html', 
                prediction=class_name, 
                image_path=image_data, 
                username=session.get('username'),
                confidence=f"{confidence:.2%}")

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Export the app for Vercel serverless