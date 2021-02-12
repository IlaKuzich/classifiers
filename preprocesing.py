import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


class IdentityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, input_array, y=None):
        return self

    def transform(self, input_array, y=None):
        return input_array


def group_education(df):
    df['education'] = np.where(df['education'] == 'basic.9y', 'Basic', df['education'])
    df['education'] = np.where(df['education'] == 'basic.6y', 'Basic', df['education'])
    df['education'] = np.where(df['education'] == 'basic.4y', 'Basic', df['education'])

    return df


def categorical_one_hot_encoding(df, identity_columns, categorical_columns):
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    processor = ColumnTransformer(
        transformers=[
            ('identity', IdentityTransformer(), identity_columns),
            ('transformer', categorical_transformer, categorical_columns)
        ]
    )
    matrix = processor.fit_transform(X=df)
    preprocessed_data = pd.DataFrame(matrix.toarray())

    return preprocessed_data