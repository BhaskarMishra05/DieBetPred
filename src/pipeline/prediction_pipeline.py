import pandas as pd
import numpy as np
import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object, save_object
from pandas import DataFrame
import joblib
import pickle
class PredictionPipeline():
    def __init__(self):
        self.model_path_configure = 'artifacts/ensemble_config.pkl'
        self.preprocessing_file_path = 'artifacts/preprocessing.pkl'

    def predict(self , feature: DataFrame): 
        feature = feature.copy()
        preprocessor_loaded_object = load_object(self.preprocessing_file_path)
        model_config = load_object(self.model_path_configure)

        data_transformed = preprocessor_loaded_object.transform(feature)
        pred = model_config.predict(data_transformed)
        return pred

class Prediction_Pipeline_Interacting_Class():
    def __init__(self,family_history_diabetes,
                physical_activity_minutes_per_week,
                age,
                bmi,
                triglycerides,
                ldl_cholesterol,
                systolic_bp,
                diet_score,
                waist_to_hip_ratio,hdl_cholesterol):
        self.family_history_diabetes = family_history_diabetes
        self.physical_activity_minutes_per_week = physical_activity_minutes_per_week
        self.age = age
        self.bmi = bmi
        self.triglycerides = triglycerides
        self.ldl_cholesterol = ldl_cholesterol
        self.systolic_bp = systolic_bp
        self.diet_score = diet_score
        self.waist_to_hip_ratio = waist_to_hip_ratio
        self.hdl_cholesterol = hdl_cholesterol

    def to_dataframe(self) -> DataFrame:
        data = {
                'family_history_diabetes': self.family_history_diabetes,
                'physical_activity_minutes_per_week' : self.physical_activity_minutes_per_week,
                'age' : self.age,
                'bmi' : self.bmi,
                'triglycerides' : self.triglycerides,
                'ldl_cholesterol' : self.ldl_cholesterol,
                'systolic_bp' : self.systolic_bp,
                'diet_score' : self.diet_score,
                'waist_to_hip_ratio' : self.waist_to_hip_ratio,
                'hdl_cholesterol' : self.hdl_cholesterol
        }
        return pd.DataFrame([data])
    