import sys


# for saving the pipeline
import joblib

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from conf.local import config
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

import input.preprocessors as pp


X_train = pd.read_csv("data/02_model_input/train.csv")

facebook_pipe = Pipeline([
    ('drop_duplicates', pp.DropDuplicates(config.DUPLICATED_VARS)),
    ('date_transformer', pp.DateTransformer(config.DATE_COL)),
    ('cat_encoder',pp.CategoricalEncoder(config.CAT_VARS)),
    ('drop_features', pp.DropFeatures(config.DROP_VARS)),
    ('feature_selector', pp.SelectFeatures(config.FEATURES)),
    ('scaler', pp.Scaler()),
    ('kmeans', KMeans(n_clusters=4, random_state=1961))
])

result_adjust_pipe = Pipeline([
    ('drop_duplicates', pp.DropDuplicates(config.DUPLICATED_VARS)),
    ('date_transformer', pp.DateTransformer(config.DATE_COL)),
    ('cat_encoder',pp.CategoricalEncoder(config.CAT_VARS)),
])

cluster = facebook_pipe.fit_predict(X_train)
X_train_new = result_adjust_pipe.fit_transform(X_train)

X_train_new = X_train_new[config.FEATURES]
X_train_new.loc[:, "Cluster"] = cluster

sum_groups = X_train_new.groupby("Cluster").sum()

joblib.dump(facebook_pipe, 'data/03_models/facebook_predict_pipeline.joblib')
joblib.dump(result_adjust_pipe, 'data/03_models/facebook_preprocess_pipeline.joblib')

