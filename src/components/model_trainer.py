import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    ExtraTreesRegressor
)
from sklearn.metrics import r2_score

from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join(
        "artifacts",
        "model.pkl"
    )


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):

        try:

            logging.info("Splitting training and testing arrays")

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            models = {

                "Linear Regression": LinearRegression(),

                "Decision Tree": DecisionTreeRegressor(random_state=42),

                "Random Forest": RandomForestRegressor(random_state=42),

                "Extra Trees": ExtraTreesRegressor(random_state=42),

                "Gradient Boosting": GradientBoostingRegressor(random_state=42),

                "AdaBoost": AdaBoostRegressor(random_state=42),

                "XGBoost": XGBRegressor(
                    random_state=42,
                    objective="reg:squarederror"
                ),

                "CatBoost": CatBoostRegressor(
                    verbose=False,
                    random_state=42
                )
            }

            params = {

                "Linear Regression": {},

                "Decision Tree": {
                    "max_depth": [5, 10, 15, 20, None],
                    "min_samples_split": [2, 5, 10]
                },

                "Random Forest": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [10, 20, 30, None],
                    "min_samples_split": [2, 5, 10]
                },

                "Extra Trees": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [10, 20, None]
                },

                "Gradient Boosting": {
                    "learning_rate": [0.01, 0.05, 0.1],
                    "n_estimators": [100, 200],
                    "subsample": [0.8, 1.0]
                },

                "AdaBoost": {
                    "learning_rate": [0.01, 0.1, 1],
                    "n_estimators": [50, 100, 200]
                },

                "XGBoost": {
                    "learning_rate": [0.01, 0.05, 0.1],
                    "n_estimators": [100, 200, 300],
                    "max_depth": [3, 5, 7]
                },

                "CatBoost": {
                    "depth": [4, 6, 8],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [100, 200]
                }
            }

            logging.info("Starting model evaluation")

            model_report, best_models = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params
            )

            best_model_name = max(
                model_report,
                key=model_report.get
            )

            best_model_score = model_report[best_model_name]

            best_model = best_models[best_model_name]

            logging.info(f"Best Model : {best_model_name}")
            logging.info(f"Best R2 Score : {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predictions = best_model.predict(X_test)

            r2_square = r2_score(y_test, predictions)

            print("\n")
            print("=" * 60)
            print(f"Best Model : {best_model_name}")
            print(f"Test R2 Score : {r2_square:.4f}")
            print("=" * 60)

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
        

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":

    ingestion = DataIngestion()

    train_path, test_path = ingestion.initiate_data_ingestion()

    print("Data Ingestion Completed Successfully")

    transformation = DataTransformation()

    train_arr, test_arr, _ = transformation.initiate_data_transformation(
        train_path,
        test_path
    )

    print("Data Transformation Completed Successfully")

    trainer = ModelTrainer()

    r2 = trainer.initiate_model_trainer(
        train_arr,
        test_arr
    )

    print("Model Training Completed Successfully")
    print(f"Final R2 Score : {r2:.4f}")
