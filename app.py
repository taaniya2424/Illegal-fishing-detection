from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open("fishing_model.pkl", "rb") as f:
    model = pickle.load(f)

# Predefined data for oceans and countries (approximate ranges)
ocean_regions = {
    "Pacific Ocean": {
        "lat_range": (lambda lat: -60 <= lat <= 60),
        "long_range": (lambda long: 100 <= long <= 290 or -180 <= long <= -70)
    },
    "Atlantic Ocean": {
        "lat_range": (lambda lat: -60 <= lat <= 60),
        "long_range": (lambda long: -70 <= long <= 20)
    },
    "Indian Ocean": {
        "lat_range": (lambda lat: -60 <= lat <= 30),
        "long_range": (lambda long: 20 <= long <= 100)
    },
    "Arctic Ocean": {
        "lat_range": (lambda lat: 60 <= lat <= 90),
        "long_range": (lambda long: -180 <= long <= 180)
    },
    "Southern Ocean": {
        "lat_range": (lambda lat: -90 <= lat <= -60),
        "long_range": (lambda long: -180 <= long <= 180)
    }
}

# Simplified country data (approximate centroids and ranges)
country_data = {
    "Japan": {"lat": 36.2048, "long": 138.2529, "range": 10},
    "USA": {"lat": 37.0902, "long": -95.7129, "range": 15},
    "Australia": {"lat": -25.2744, "long": 133.7751, "range": 15},
    "India": {"lat": 20.5937, "long": 78.9629, "range": 10},
    "Brazil": {"lat": -14.2350, "long": -51.9253, "range": 15}
}

def get_ocean(latitude, longitude):
    for ocean, ranges in ocean_regions.items():
        if ranges["lat_range"](latitude) and ranges["long_range"](longitude):
            return ocean
    return "Unknown Ocean"

def get_nearest_country(latitude, longitude):
    min_distance = float('inf')
    nearest_country = "Unknown Country"
    
    for country, data in country_data.items():
        lat_diff = abs(data["lat"] - latitude)
        long_diff = abs(data["long"] - longitude)
        distance = (lat_diff ** 2 + long_diff ** 2) ** 0.5  # Euclidean distance
        
        if distance <= data["range"] and distance < min_distance:
            min_distance = distance
            nearest_country = country
    
    return nearest_country

def get_reasoning(latitude, longitude, speed, proximity, prediction):
    reasons = []
    if prediction == "Yes":
        if proximity < 2:
            reasons.append("Proximity to protected area is too close (< 2 nautical miles), violating conservation laws.")
        if speed > 15:
            reasons.append("Vessel speed is excessively high (> 15 knots), suggesting potential illegal activity.")
        if latitude > 60 or latitude < -60:
            reasons.append("Location is in a polar region (Arctic or Southern Ocean), where fishing is heavily regulated or prohibited.")
        if not reasons:
            reasons.append("The combination of location, speed, and proximity indicates a high likelihood of illegal fishing based on the model.")
    else:
        if proximity > 5:
            reasons.append("Safe distance from protected areas (> 5 nautical miles) suggests legal operation.")
        if speed < 10:
            reasons.append("Moderate vessel speed (< 10 knots) is consistent with legal fishing practices.")
        if -60 <= latitude <= 60 and -70 <= longitude <= 100:
            reasons.append("Location is within a commonly fished ocean region with fewer restrictions.")
        if not reasons:
            reasons.append("The combination of location, speed, and proximity indicates a low likelihood of illegal fishing based on the model.")
    return "; ".join(reasons)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    latitude = float(request.form['latitude'])
    longitude = float(request.form['longitude'])
    speed = float(request.form['speed'])
    proximity = float(request.form['proximity'])

    # Determine ocean and country based on latitude and longitude
    ocean = get_ocean(latitude, longitude)
    country = get_nearest_country(latitude, longitude)

    # Prepare input for the model
    input_data = np.array([[latitude, longitude, speed, proximity]])
    prediction_proba = model.predict_proba(input_data)
    probability = f"{max(prediction_proba[0]) * 100:.0f}%"
    prediction = "Yes" if model.predict(input_data)[0] == 1 else "No"

    # Get reasoning for the prediction
    reasoning = get_reasoning(latitude, longitude, speed, proximity, prediction)

    return render_template('result.html', 
                          prediction=prediction, 
                          probability=probability, 
                          ocean=ocean, 
                          country=country,
                          reasoning=reasoning)

if __name__ == '__main__':
    app.run(debug=True)