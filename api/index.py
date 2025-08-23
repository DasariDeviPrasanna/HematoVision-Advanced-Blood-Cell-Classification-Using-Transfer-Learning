from flask import Flask, request, jsonify
import os

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({
        "message": "Blood Cell Classification API",
        "status": "running",
        "endpoints": ["/", "/health", "/predict"]
    })

@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Simple mock prediction
        import random
        labels = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']
        prediction = random.choice(labels)
        confidence = round(random.uniform(0.7, 0.95), 2)
        
        return jsonify({
            "success": True,
            "prediction": prediction,
            "confidence": confidence,
            "filename": file.filename
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "type": "prediction_error"
        }), 500

@app.route('/test')
def test():
    return jsonify({"message": "API is working!"})

# Vercel handler
def handler(request, context):
    return app(request, context)

if __name__ == '__main__':
    app.run(debug=True)
