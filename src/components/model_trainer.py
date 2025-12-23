import pandas as pd
import numpy as np
import os
import sys
import pickle
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, load_object
from sklearn.linear_model import LogisticRegression as LR
from xgboost import XGBClassifier as XGBC
from lightgbm import LGBMClassifier as LGBMC
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from dataclasses import dataclass

logging.info("MODEL TRAINER SCRIPT LOADED")

@dataclass
class MODEL_TRAINER_CONFIG:
    lr_model_path = os.path.join("artifacts", "lr_model.pkl")
    xgb_model_path = os.path.join("artifacts", "xgb_model.pkl")
    lgbm_model_path = os.path.join("artifacts", "lgbm_model.pkl")
    ensemble_config_path = os.path.join("artifacts", "ensemble_config.pkl")

class MODEL_TRAINER:
    def __init__(self):
        logging.info("INITIALIZING MODEL TRAINER")
        self.model_trainer_config_object = MODEL_TRAINER_CONFIG()

    def model_trainer_main_training_method(self, training_array, testing_array):
        try:
            logging.info("SPLITTING TRAIN AND TEST ARRAYS")

            X_train = training_array[:, :-1]
            y_train = training_array[:, -1].ravel()

            X_test = testing_array[:, :-1]
            y_test = testing_array[:, -1].ravel()

            logging.info("INITIALIZING MODELS")

            model_xgb = XGBC(use_label_encoder=False, eval_metric="logloss")
            model_lgbm = LGBMC()
            model_lr = LR(max_iter=1000)

            logging.info("TRAINING XGBOOST MODEL")
            model_xgb.fit(X_train, y_train)

            logging.info("TRAINING LIGHTGBM MODEL")
            model_lgbm.fit(X_train, y_train)

            logging.info("TRAINING LOGISTIC REGRESSION MODEL")
            model_lr.fit(X_train, y_train)

            logging.info("GENERATING PREDICTIONS")

            y_pred_xgb = model_xgb.predict_proba(X_test)[:, 1]
            y_pred_lgbm = model_lgbm.predict_proba(X_test)[:, 1]
            y_pred_lr = model_lr.predict_proba(X_test)[:, 1]

            logging.info("SAVING TRAINED MODELS")

            save_object(self.model_trainer_config_object.lr_model_path, model_lr)
            save_object(self.model_trainer_config_object.xgb_model_path, model_xgb)
            save_object(self.model_trainer_config_object.lgbm_model_path, model_lgbm)

            logging.info("MODEL TRAINING COMPLETED")

            return y_pred_lr, y_pred_lgbm, y_pred_xgb, y_test

        except Exception as e:
            raise CustomException(e, sys)

    def weighted_avg(self,y_test,prediction_lr,prediction_lgbm,prediction_xgb,weight_lr: float,weight_xgb: float,weight_lgbm: float):
        try:
            logging.info("STARTING WEIGHTED ENSEMBLE PREDICTION")

            ensemble_weights = {
                "weight_lr": weight_lr,
                "weight_xgb": weight_xgb,
                "weight_lgbm": weight_lgbm
            }

            save_object(self.model_trainer_config_object.ensemble_config_path,ensemble_weights)

            weighted_score = (
                prediction_lr * weight_lr +
                prediction_xgb * weight_xgb +
                prediction_lgbm * weight_lgbm
            )

            final_prediction = (weighted_score >= 0.5).astype(int)

            acc = accuracy_score(y_test, final_prediction)
            prec = precision_score(y_test, final_prediction)
            rec = recall_score(y_test, final_prediction)
            cm = confusion_matrix(y_test, final_prediction)

            logging.info(f"ENSEMBLE ACCURACY: {acc}")
            logging.info(f"ENSEMBLE PRECISION: {prec}")
            logging.info(f"ENSEMBLE RECALL: {rec}")
            logging.info(f"CONFUSION MATRIX:\n{cm}")

            return final_prediction

        except Exception as e:
            raise CustomException(e, sys)
