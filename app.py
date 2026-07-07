from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load the pre-trained model
try:
    model = joblib.load("model.pkl")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# A map of crop labels to their respective emojis for a richer UI experience
CROP_EMOJIS = {
    'rice': '🌾 Rice',
    'maize': '🌽 Maize',
    'chickpea': '🫘 Chickpea',
    'kidneybeans': '🫘 Kidney Beans',
    'pigeonpeas': '🫛 Pigeon Peas',
    'mothbeans': '🫘 Moth Beans',
    'mungbean': '🫛 Mung Bean',
    'blackgram': '🫘 Black Gram',
    'lentil': '🍲 Lentil',
    'pomegranate': '🍎 Pomegranate',
    'banana': '🍌 Banana',
    'mango': '🥭 Mango',
    'grapes': '🍇 Grapes',
    'watermelon': '🍉 Watermelon',
    'muskmelon': '🍈 Muskmelon',
    'apple': '🍎 Apple',
    'orange': '🍊 Orange',
    'papaya': '🥭 Papaya',
    'coconut': '🥥 Coconut',
    'cotton': '☁️ Cotton',
    'jute': '🌱 Jute',
    'coffee': '☕ Coffee'
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model is not loaded."}), 500
    
    try:
        # Extract features from form input
        n = float(request.form.get('N', 0))
        p = float(request.form.get('P', 0))
        k = float(request.form.get('K', 0))
        temp = float(request.form.get('temperature', 0))
        humidity = float(request.form.get('humidity', 0))
        ph = float(request.form.get('ph', 0))
        rainfall = float(request.form.get('rainfall', 0))
        
        # Prepare inputs as a dataframe to keep column names (matching the training dataframe)
        features = pd.DataFrame([[n, p, k, temp, humidity, ph, rainfall]], 
                                columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
        
        # Predict class
        prediction = model.predict(features)[0]
        
        # Calculate confidence score
        probabilities = model.predict_proba(features)[0]
        max_prob_idx = np.argmax(probabilities)
        confidence = float(probabilities[max_prob_idx]) * 100
        
        # Get user-friendly name and emoji
        recommended_crop = CROP_EMOJIS.get(prediction.lower(), prediction.capitalize())
        
        return jsonify({
            "success": True,
            "crop": recommended_crop,
            "confidence": f"{confidence:.1f}%",
            "raw_crop": prediction
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
