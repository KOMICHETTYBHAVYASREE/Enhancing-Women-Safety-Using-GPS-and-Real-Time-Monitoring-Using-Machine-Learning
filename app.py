from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load trained model and feature columns
model = joblib.load('latest_rf_model.pkl')  # Load the trained model
feature_columns = joblib.load('feature_columns.pkl')  # Load feature names

# Cities encoded mapping (Ensure you have a mapping for city encoding)
city_mapping = {
    'Agra': 0, 'Ahmedabad': 1, 'Bangalore': 2, 'Bhopal': 3, 'Chennai': 4, 'Delhi': 5, 'Faridabad': 6, 'Ghaziabad': 7,
    'Hyderabad': 8, 'Indore': 9, 'Jaipur': 10, 'Kalyan': 11, 'Kanpur': 12, 'Kolkata': 13, 'Lucknow': 14, 'Ludhiana': 15,
    'Meerut': 16, 'Mumbai': 17, 'Nagpur': 18, 'Nashik': 19, 'Patna': 20, 'Pune': 21, 'Rajkot': 22, 'Srinagar': 23,
    'Surat': 24, 'Thane': 25, 'Varanasi': 26, 'Vasai': 27, 'Visakhapatnam': 28
}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    crime_description = request.form['crime_description']
    crime_domain = request.form['crime_domain']
    police_deployed = int(request.form['police_deployed'])
    days_to_close_cases = int(request.form['days_to_close_cases'])
    days_taken_to_report = int(request.form['days_taken_to_report'])

    city = request.form['city']
    victim_gender = request.form['victim_gender']
    victim_gender_f = 1 if victim_gender == 'F' else 0
    victim_gender_m = 1 if victim_gender == 'M' else 0
    victim_gender_x = 1 if victim_gender == 'X' else 0

    case_closed_no = 1 if 'case_closed_no' in request.form else 0


    # Encode categorical features
    le = LabelEncoder()

    crime_description_encoded = le.fit_transform([crime_description])[0]
    crime_domain_encoded = le.fit_transform([crime_domain])[0]
    city_encoded = city_mapping.get(city, -1)  # Get encoded value for the city, default to -1 if not found

    # Prepare the input data as a DataFrame
    data = pd.DataFrame({
        'Crime Description': [crime_description_encoded],
        'Crime Domain': [crime_domain_encoded],
        'Police Deployed': [police_deployed],
        'Days_to_close_cases': [days_to_close_cases],
        'Days_taken_to_report': [days_taken_to_report],
        'City_Encoded': [city_encoded],
        'Victim Gender_F': [victim_gender_f],
        'Victim Gender_M': [victim_gender_m],
        'Victim Gender_X': [victim_gender_x],
        'Case Closed_No': [case_closed_no]
    })

    # Make prediction using the trained model
    prediction = model.predict(data)
    # Make prediction using the trained model
    
    probability = model.predict_proba(data)[0]
    # Map prediction result to the corresponding risk level
    risk_level = "Low" if prediction[0] == 0 else "Medium" if prediction[0] == 1 else "High"

    # Render the output template with the prediction result
    probability_low = "{:.2f}%".format(probability[0] * 100)
    probability_medium = "{:.2f}%".format(probability[1] * 100)


    return render_template('output.html', city=city, risk_level=risk_level, 
                         probability_low=probability_low, probability_medium=probability_medium)

@app.route('/index')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
