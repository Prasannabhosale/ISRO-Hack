import os
import sys
import pickle

from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV

from src.exception import CustomException


def save_object(file_path, obj):
    """
    Saves a Python object (model, preprocessor, etc.) to the specified file path.
    """
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Loads a saved Python object.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Performs hyperparameter tuning using RandomizedSearchCV,
    evaluates all models using R² score,
    and returns:
        1. report (model name -> test R² score)
        2. best_models (model name -> trained model)
    """

    try:

        report = {}
        best_models = {}

        for model_name, model in models.items():

            print(f"\n{'='*60}")
            print(f"Training : {model_name}")
            print(f"{'='*60}")

            parameters = param.get(model_name, {})

            if parameters:

                random_search = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=parameters,
                    n_iter=10,
                    cv=5,
                    scoring="r2",
                    random_state=42,
                    n_jobs=-1
                )

                random_search.fit(X_train, y_train)

                best_model = random_search.best_estimator_

                print("Best Parameters:", random_search.best_params_)

            else:

                best_model = model
                best_model.fit(X_train, y_train)

            # Predictions
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            # R² Scores
            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)

            print(f"Train R² Score : {train_score:.4f}")
            print(f"Test  R² Score : {test_score:.4f}")

            report[model_name] = test_score
            best_models[model_name] = best_model

        return report, best_models

    except Exception as e:
        raise CustomException(e, sys)
