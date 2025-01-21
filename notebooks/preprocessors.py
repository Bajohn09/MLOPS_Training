import pandas as pd
import numpy as np
import scipy.stats as stats

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin

class DropDuplicates(BaseEstimator, TransformerMixin):
    def __init__(self, variables):

        self.variables = variables

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        return self

    def transform(self, X):

    	# Do not over-write original dataframe
        X = X.copy(deep=True)
        X = X.drop_duplicates(subset=self.variables, keep='first')

        return X

class YeoJohnsonTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, variables):

        self.variables = variables

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        return self

    def transform(self, X):

    	# Do not over-write original dataframe
        X = X.copy(deep=True)

        for num_col in self.variables:
            X[num_col], _ = stats.yeojohnson(X[num_col])

        return X
    
class DateTransformer(BaseEstimator, TransformerMixin):
	# Temporal elapsed time transformer

    def __init__(self, date_variable):

        self.date_variable = date_variable[0]

    def fit(self, X, y=None):
        

        # we need this step to fit the sklearn pipeline
        return self

    def transform(self, X):

    	# Do not over-write original dataframe
        X = X.copy(deep=True)
        X[self.date_variable] = pd.to_datetime(X[self.date_variable])

        X["day"] = X[self.date_variable].dt.day
        X["weekday"] = X[self.date_variable].dt.weekday
        X["month"] = X[self.date_variable].dt.month
        X["hour"] = X[self.date_variable].dt.hour

        return X

class IDTransformer(BaseEstimator, TransformerMixin):
	# Temporal elapsed time transformer

    def __init__(self, status_variable):

        self.status_variable = status_variable[0]

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        return self

    def transform(self, X):

    	# Do not over-write original dataframe
        X = X.copy(deep=True)
        X["event"] = X[self.status_variable].apply(lambda x: x.split("_")[0])
        
        return X
    
class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cat_variables):
        self.cat_variables = cat_variables

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        return self

    def transform(self, X):

    	# Do not over-write original dataframe
        X = X.copy(deep=True)

        encoder = OneHotEncoder(drop=None, sparse_output=False)
        encoded_data = encoder.fit_transform(X[self.cat_variables])
        
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(self.cat_variables))
        
        # Drop original columns and concatenate the encoded ones
        X = X.drop(columns=self.cat_variables).reset_index(drop=True)
        full_data =  pd.concat([X, encoded_df], axis=1)
        
        return full_data

class DropFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, drop_variables):
        self.drop_variables = drop_variables

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        return self

    def transform(self, X):

    	# Do not over-write original dataframe
        X = X.copy(deep=True)
        X = X.drop(self.drop_variables, axis=1)

        return X


class SelectFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, select_variables):
        self.select_variables = select_variables

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        return self

    def transform(self, X):

    	# Do not over-write original dataframe
        X = X.copy(deep=True)
        X = X.loc[:, self.select_variables]

        return X


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, dataframe):
        self.df = dataframe

    def calculate_iterations(
        self, n_features, base_iterations=100, multiplier=50
    ):
        """
        Calculate the number of iterations based on the number of features.
        """
        n_iterations = base_iterations + multiplier * np.log(n_features)
        return int(n_iterations)

    def tree_based_feature_importance(self, n_select_features, rf_params=None):
        """
        Unsupervised Feature Selection with Random Forests (UFSRF)
        """
        # Number of features to select
        n_features = n_select_features

        # Calculate the number of iterations based on the total features
        n_iterations = self.calculate_iterations(self.df.shape[1])

        # Placeholder for feature importances across all iterations
        aggregated_importances = np.zeros(self.df.shape[1])

        # Ensure all data is float type
        df_float = self.df.astype(float)

        # Setting default RandomForest parameters if none provided
        if rf_params is None:
            rf_params = {
                "n_estimators": 150,
                "max_depth": 10,
                "random_state": 42,
            }

        for _ in range(n_iterations):
            # Generating random discrete class labels for classification
            random_labels = np.random.randint(0, 2, df_float.shape[0])
            model = RandomForestClassifier(**rf_params)
            model.fit(df_float, random_labels)
            aggregated_importances += model.feature_importances_

        # Averaging feature importances over all iterations
        aggregated_importances /= n_iterations

        # Sorting features based on their aggregated importance scores
        important_features = df_float.columns[
            np.argsort(aggregated_importances)[-n_features:]
        ]

        return df_float[important_features]
    
class Scaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        return self

    def transform(self, X):

    	# Do not over-write original dataframe
        X = X.copy(deep=True)
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(X)
        scale_df = pd.DataFrame(scaled_data, columns=scaler.get_feature_names_out())
        
        return scale_df