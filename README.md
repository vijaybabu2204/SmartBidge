# 🌱 OptiCrop: Smart Agricultural Production Optimization Engine

OptiCrop is a state-of-the-art agricultural recommendation platform that leverages Machine Learning to help farmers identify the ideal crop to grow based on soil nutrients and meteorological conditions.

🔗 **Live Deployment:** [http://VijayBabu22.pythonanywhere.com](http://VijayBabu22.pythonanywhere.com)

---

## 🎯 Key Project Features

- **Dashboard Layout:** Floating modern navigation bar supporting smooth view transitions across Home, About, and FindYourCrop tabs.
- **Synced Calibration Sliders:** Interactive range sliders bound with numeric inputs allowing quick soil profile settings.
- **Nutrient Balance Visualization:** Real-time CSS N-P-K (Nitrogen, Phosphorus, Potassium) soil graph representing active macronutrient percentages.
- **Circular SVG Match Gauge:** Custom SVG progress ring presenting model classification confidence with dynamic stroke offset animations.
- **Soil Assessments:** Evaluates outputs and serves detailed explanations behind recommended crops.
- **Simulation Presets:** One-click shortcuts for testing typical conditions (Rice, Coffee, Watermelon, Banana).

---

## 🧠 Machine Learning Model

The predictive engine is built using **Random Forest Classifier** trained on the agricultural datasets:
- **Dataset Size:** 2200 rows of environmental data
- **Inputs Analysed:** Nitrogen (N), Phosphorus (P), Potassium (K), Temperature, Humidity, pH level, and Rainfall
- **Target Target Output:** Crop classification label (Rice, Maize, Coffee, etc.)
- **Model Accuracy:** **99.3%** on test evaluation
- **Saved Model File:** `model.pkl`

---

## 📁 Repository Structure

```text
├── Crop_recommendation.csv   # Model training dataset
├── train_model.py            # Model training & verification script
├── app.py                    # Flask server backend logic
├── model.pkl                 # Trained Random Forest classifier binary
├── requirements.txt          # Python dependencies
├── render.yaml               # Render Blueprint deploy specs
├── .gitignore                # Exclude runtime caches & virtualenvs
├── static/
│   └── images/
│       └── farmer.jpg        # About page visual asset
└── templates/
    └── index.html            # Main UI Dashboard single-page application
```

---

## 💻 Local Setup & Installation

Follow these commands to configure the workspace on your machine:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/vijaybabu2204/SmartBidge.git
   cd SmartBidge
   ```

2. **Setup virtual environment & dependencies:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/MacOS:
   source .venv/bin/activate

   pip install -r requirements.txt
   ```

3. **Train the model (Optional):**
   ```bash
   python train_model.py
   ```

4. **Launch Flask application:**
   ```bash
   python app.py
   ```
   Open your browser and navigate to `http://127.0.0.1:5000`.
