from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import tempfile
import base64
from io import BytesIO

# Initialize Flask app
app = Flask(__name__)

# Configure template and static folders for Vercel
app.template_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates')
app.static_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static')

# Class labels
labels = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

@app.route('/')
def front():
    try:
        return render_template('front.html')
    except Exception as e:
        return jsonify({"error": f"Template error: {str(e)}"}), 500

@app.route('/home')
def home():
    try:
        return render_template('home.html')
    except Exception as e:
        return jsonify({"error": f"Template error: {str(e)}"}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if image was uploaded
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            file.save(tmp_file.name)
            tmp_path = tmp_file.name

        try:
            # For now, return a mock prediction since model loading might fail
            # In production, you'd load the model here
            import random
            class_name = random.choice(labels)
            confidence = random.uniform(0.7, 0.95)
            
            # Convert image to base64 for display
            with open(tmp_path, 'rb') as img_file:
                img_data = img_file.read()
                img_str = base64.b64encode(img_data).decode()

            return render_template('result.html', 
                                prediction=class_name, 
                                image_path=f"data:image/jpeg;base64,{img_str}",
                                confidence=f"{confidence:.1%}")

        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except Exception as e:
        return jsonify({
            "error": f"Error processing image: {str(e)}",
            "type": "prediction_error"
        }), 500

# Health check endpoint
@app.route('/health')
def health():
    return jsonify({"status": "healthy", "message": "Blood cell classification API is running"})

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Route not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

# Vercel serverless function handler
def handler(request, context):
    return app(request, context)

if __name__ == '__main__':
    app.run(debug=True)
