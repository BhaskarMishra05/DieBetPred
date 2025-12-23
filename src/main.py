from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DATA_INGESTION
from src.components.data_transformation import DATA_TRANSFORMATION
from src.components.model_trainer import MODEL_TRAINER
import sys
import os

try:
    ingestion_objects = DATA_INGESTION()
    _, train, test = ingestion_objects.ingestion_method_for_data_spliting()
except Exception as e:
    raise CustomException(e, sys)

try:
    transformation_object = DATA_TRANSFORMATION()
    train_array, test_array = transformation_object.transformation_method(
        train= train, test= test
    )
except Exception as e:
    raise CustomException(e,sys)

try:
    model_trainer_object = MODEL_TRAINER()
    y_pred_lr, y_pred_lgbm , y_pred_xgb ,y_test = model_trainer_object.model_trainer_main_training_method(
        training_array= train_array, testing_array= test_array)
    final_prediction = model_trainer_object.weighted_avg(y_test, prediction_lr= y_pred_lr,
                                                         prediction_lgbm= y_pred_lgbm,
                                                         prediction_xgb= y_pred_xgb,
                                                         weight_lgbm= 0.3,
                                                         weight_xgb= 0.4,
                                                         weight_lr= 0.3)
except Exception as e:
    raise CustomException(e,sys)