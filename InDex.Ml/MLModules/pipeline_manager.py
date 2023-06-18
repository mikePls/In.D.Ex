import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler, LabelEncoder, OneHotEncoder, \
    OrdinalEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, ElasticNet
from sklearn.model_selection import KFold, cross_val_predict, GridSearchCV, train_test_split


class PipelineManager:

    def generate_pipeline(self, model:str=None, transformer:ColumnTransformer=None, params:dict=None):
        if not model:
            return
        models = {'Logistic regression':LogisticRegression,
                  'Neural network (MLP Classifier)':MLPClassifier,
                  'Elastic net':ElasticNet,
                  'Linear regression':LinearRegression}
        steps = []
        selected_model = models[model](**params) if params is not None else models[model]()
        if transformer is not None and model in models.keys():
            steps.append(('preprocessor',transformer))
            steps.append((model, selected_model))
            return Pipeline(steps=steps)
        elif not transformer:
            return Pipeline(steps=[(model, selected_model)])


    def generate_column_transformer(self, cat_cols:list=None, num_cols:list=None, cat_transformer:str=None, num_scaler:str=None):
        transformer_options = {'Normal scaler (min-max)':MinMaxScaler(),
                               'Standard scaler (mean:0, std:1)':StandardScaler(),
                               'Label encoder':OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
                               'One-hot encoder':OneHotEncoder(handle_unknown='ignore')}
        transformers=[]
        if cat_cols is not None and cat_transformer is not None:
            transformers.append(('categorical_cols', transformer_options[cat_transformer], cat_cols))
        if num_cols is not None and num_scaler is not None:
            transformers.append(('numerical_cols', transformer_options[num_scaler], num_cols))
        if len(transformers)>0:
            col_transformer = ColumnTransformer(transformers=transformers, remainder='passthrough')
            return col_transformer
        else:
            return None

    def fit_pipeline(self, pipeline:Pipeline, x, y):
        return pipeline.fit(X=x, y=y)

    def grid_search_cv(self, pipeline:Pipeline, param_grid:dict, x, y, cv=5)->GridSearchCV:
        """Takes a pipeline object, a parameter grid with relevant hyperparameters, x features and y
        pandas Series as a target variable.
        Generates a GridSearchCV object from sklearn and fits it with x and y. Returns the fitted object"""
        grid_search:GridSearchCV = GridSearchCV(pipeline, param_grid, cv=cv)
        grid_search.fit(x , y)
        return grid_search

    def train_test_split(self, test_size:float, x:DataFrame, y:pd.Series, random_state:int=42):
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test

    def __show_loading_window(self):
        pass

    def get_model_type(self, model):
        """
        Determines the type of machine learning model and returns
        the relevant metrics for that type.

        Parameters:
        -- model: The machine learning model object.

        Returns:
        model_type (str): The type of the model ('regression', 'classification', or 'unknown').
        """
        if hasattr(model, 'predict') and hasattr(model, 'score'): # determine if object is ml model
            if callable(model.predict) and callable(model.score): # determine if model is fitted
                if hasattr(model, 'predict_proba'): # if it can predict probability it's a classification model
                    if callable(model.predict_proba): # determine if model is classification model
                        return 'classification'
                else:
                    return 'regression'
            else:
                return 'error: model not trained'
        return 'unknown', None

    def get_model_metrics(self, pipeline, X_test, y_test)->str:
        if X_test is not None: #if a test sample is available
            # determine the type of model, then retrieve relevant metrics
            _type = self.get_model_type(pipeline)
            if _type == 'classification':
                return self.__get_classification_metrics(pipeline, X_test, y_test)
            elif _type == 'regression':
                return self.__get_regression_metrics(pipeline, X_test, y_test)
            elif _type == 'unknown':
                return 'Model error: model not trained, or of unknown type'
            elif X_test is None or y_test is None:
                return 'No test sample was selected.'


    def __get_regression_metrics(self, pipeline:Pipeline, X_test, y_test)->str:
        """
            Retrieves relevant regression metrics for a regression model.

            Args:
                y_true (array-like): True labels.
                y_pred (array-like): Predicted labels.

            Returns:
                str: A formatted string containing the following metrics:
                    - Mean Squared Error (MSE)
                    - Mean Absolute Error (MAE)
                    - R-squared (R2) score
            """
        y_pred = pipeline.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics_str = f"Mean Squared Error (MSE): {mse:.4f}\n"
        metrics_str += f"Mean Absolute Error (MAE): {mae:.4f}\n"
        metrics_str += f"R-squared (R2) score: {r2:.4f}"

        return metrics_str

    def __get_classification_metrics(self, model, X_test, y_test)->str:
        metrics = {}
        y_pred: BaseEstimator = model.predict(X_test)
        metrics['accuracy'] = accuracy_score(y_test, y_pred)

        # Precision
        metrics['precision'] = precision_score(y_test, y_pred, average='weighted')

        # Recall
        metrics['recall'] = recall_score(y_test, y_pred, average='weighted')

        # F1-score
        metrics['f1_score'] = f1_score(y_test, y_pred, average='weighted')

        # Confusion Matrix
        metrics['confusion_matrix'] = self.__get_confusion_matrix(y_test, y_pred)

        result = ""
        for key, value in metrics.items():
            row = f"{key}: {value}\n"  # Concatenate key and value with newline
            result += row
        return str(result)

    def __get_confusion_matrix(self, y_test, y_pred):
        """
        Compute confusion matrix and format it with labels for rows and columns.

        Parameters:
            - y_test (array-like): Ground truth labels
            - y_pred (array-like): Predicted labels

        Returns:
            str: Formatted confusion matrix string
        """
        labels = ["True Neg", "True Pos"]
        cm = confusion_matrix(y_test, y_pred)
        rows = ["\t".join([label.ljust(12)] + [str(value).rjust(10) for value in row])
                for label, row in zip(labels, cm)]
        header = "\t".join(["".ljust(12)] + [f"Predicted {i}".rjust(10) for i in range(len(cm))])
        return f"\n{header}\n" + "\n".join(rows)




