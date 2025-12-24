from flask import Flask, render_template, request
import sys
from src.pipeline.prediction_pipeline import (
    PredictionPipeline,
    Prediction_Pipeline_Interacting_Class
)
from src.exception import CustomException

app = Flask(__name__)

# Home page route
@app.route('/')
def home():
    return render_template('home.html')

# Prediction page route
@app.route('/predictions', methods=['GET', 'POST'])
def predictions():
    final_result = None  # Will hold the prediction result
    
    if request.method == 'POST':
        try:
            # Collect data from form
            data = Prediction_Pipeline_Interacting_Class(
                family_history_diabetes=int(request.form.get('family_history_diabetes')),
                physical_activity_minutes_per_week=float(request.form.get('physical_activity_minutes_per_week')),
                age=int(request.form.get('age')),
                bmi=float(request.form.get('bmi')),
                triglycerides=float(request.form.get('triglycerides')),
                ldl_cholesterol=float(request.form.get('ldl_cholesterol')),
                systolic_bp=float(request.form.get('systolic_bp')),
                diet_score=float(request.form.get('diet_score')),
                waist_to_hip_ratio=float(request.form.get('waist_to_hip_ratio')),
                hdl_cholesterol=float(request.form.get('hdl_cholesterol'))
            )
            
            input_df = data.to_dataframe()  # Convert to DataFrame
            pipeline = PredictionPipeline()  # Initialize your prediction pipeline
            prediction = pipeline.predict(input_df)  # Make prediction
            final_result = "YES" if prediction == 1 else "NO"  # Convert to human-readable
            
        except Exception as e:
            raise CustomException(e, sys)
    
    # Render the prediction page, pass final_result (None if GET request)
    return render_template('index.html', final_result=final_result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)