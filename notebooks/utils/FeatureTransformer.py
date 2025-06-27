from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class ADRThirdQuartileDeviationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.q3_map_ = None
        self.group_cols = ['DistributionChannel', 'ReservedRoomType', 'ArrivalDateYear', 'ArrivalDateWeekNumber']

    def fit(self, X, y=None):
        X = pd.DataFrame(X)  # assicurati che sia DataFrame
        self.q3_map_ = (
            X.groupby(self.group_cols)['ADR']
            .quantile(0.75)
            .reset_index()
            .rename(columns={'ADR': 'ThirdQuartileADR'})
        )
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X = X.merge(self.q3_map_, on=self.group_cols, how='left')

        # Vettorizzato (più veloce di apply)
        X['ADRThirdQuartileDeviation'] = np.where(
            X['ThirdQuartileADR'] > 0,
            X['ADR'] / X['ThirdQuartileADR'],
            0
        )

        X.drop(columns=[
            'ADR',
            'ThirdQuartileADR',
            'ArrivalDateYear',
            'ArrivalDateMonth',
            'ArrivalDateWeekNumber',
            'ArrivalDateDayOfMonth',
            'ReservedRoomType'
        ], inplace=True, errors='ignore')
        return X

    def get_feature_names_out(self, input_features=None):
        return ['ADRThirdQuartileDeviation'] 
