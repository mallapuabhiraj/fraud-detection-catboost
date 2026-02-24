from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd


class Preprocess(BaseEstimator, TransformerMixin):

    @staticmethod
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371.0
        # Use np.radians instead of math.radians for array support
        lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
        lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad
        a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        # Use np.round instead of round for array support
        return np.round(R * c)

    def preprocess(self, df):
        df = df.copy()

        # 1. Parsing dates to DateTime format
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], format='mixed')
        df['dob'] = pd.to_datetime(df['dob'], format='mixed', errors='coerce')

        # 2. Calculating age and distance features
        df['age'] = ((df['trans_date_trans_time'] - df['dob']).dt.days // 365).clip(18, 100).astype(int)
        df['distance_km'] = self.haversine_distance(
            df['lat'].values, df['long'].values,
            df['merch_lat'].values, df['merch_long'].values
        )

        df['prev_merch_lat'] = df.groupby('cc_num')['merch_lat'].shift(1)
        df['prev_merch_long'] = df.groupby('cc_num')['merch_long'].shift(1)
        df['dist_from_last_trans'] = self.haversine_distance(
            df['merch_lat'].values, df['merch_long'].values,
            df['prev_merch_lat'].fillna(df['merch_lat']).values,
            df['prev_merch_long'].fillna(df['merch_long']).values
        )

        # Velocity features
        df['trans_today'] = df.groupby(['cc_num', df['trans_date_trans_time'].dt.date]).cumcount() + 1
        df['time_since_last_min'] = (df.groupby('cc_num')['trans_date_trans_time'].diff().dt.total_seconds() / 60).fillna(1440)
        df['is_rapid_succession'] = (df['time_since_last_min'] < 5).astype(int)
        df['high_daily_activity'] = (df['trans_today'] > 3).astype(int)

        # Time since last transaction
        df['time_since_last_trans_hr'] = (df.groupby('cc_num')['trans_date_trans_time'].diff().dt.total_seconds() // 3600).fillna(24)
        df["km_per_hour"] = df["distance_km"] / (df["time_since_last_trans_hr"] + 1)
        df["is_impossible_travel"] = (df["km_per_hour"] > 500).astype(int)

        # Time and Amount features
        df['hour'] = df['trans_date_trans_time'].dt.hour
        df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
        df['month'] = df['trans_date_trans_time'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_night'] = ((df['hour'] >= 21) | (df['hour'] <= 6)).astype(int)
        df['is_midnight'] = df['hour'].between(0,3).astype(int)
        df['is_high_amount'] = (df['amt'] > 50).astype(int)  # Fixed threshold
        df['is_very_high_amount'] = (df['amt'] > 100).astype(int)  # Fixed threshold

        # Location and Age grouping
        df['is_local'] = (df['distance_km'] < 10).astype(int)
        df['is_short_travel'] = ((df['distance_km'] >= 10) & (df['distance_km'] < 100)).astype(int)
        df['is_medium_travel'] = ((df['distance_km'] >= 100) & (df['distance_km'] < 500)).astype(int)
        df['is_long_travel'] = (df['distance_km'] >= 500).astype(int)
        df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 50, 65, 100],
           labels=['young', 'young_adult', 'middle', 'senior', 'elderly'])

        # Transaction count per card
        df['trans_num_per_card'] = df.groupby('cc_num').cumcount() + 1
        df["user_night_rate"] = (
            df.groupby("cc_num")["is_night"]
            .transform(lambda x: x.expanding().mean().shift(1).fillna(0))
        )
        df["night_anomaly"] = ((df["is_night"] == 1) & (df["user_night_rate"] < 0.1)).astype(int)

        # Encoding
        df['amt_log'] = np.log1p(df['amt'])
        df['gender'] = df['gender'].map({'M': 1, 'F': 0}).fillna(0).astype(int)

        # Anomaly features
        df['amt_mean_card'] = df.groupby('cc_num')['amt'].expanding().mean().reset_index(0, drop=True)
        df['amt_std_card'] = df.groupby('cc_num')['amt'].expanding().std().reset_index(0, drop=True).fillna(1)
        df['amt_max_card'] = df.groupby('cc_num')['amt'].expanding().max().reset_index(0, drop=True)
        df['amt_zscore'] = (df['amt'] - df['amt_mean_card']) / (df['amt_std_card'] + 0.01)
        df['amt_ratio_to_max'] = df['amt'] / (df['amt_max_card'] + 1)
        df['unusual_amount'] = (df['amt_zscore'] > 2).astype(int)
        df['very_unusual_amount'] = (df['amt_zscore'] > 3).astype(int)

        df['distance_mean_card'] = df.groupby('cc_num')['distance_km'].expanding().mean().reset_index(0, drop=True)
        df['distance_std_card'] = df.groupby('cc_num')['distance_km'].expanding().std().reset_index(0, drop=True).fillna(1)
        df['distance_zscore'] = (df['distance_km'] - df['distance_mean_card']) / (df['distance_std_card'] + 0.01)
        df['unusual_location'] = (df['distance_zscore'] > 2).astype(int)

        df['prev_category'] = df.groupby('cc_num')['category'].shift(1)
        df['category_switch'] = (df['category'] != df['prev_category']).fillna(0).astype(int)

        df['merchant_first_use'] = (df.groupby(['cc_num', 'merchant']).cumcount() == 0).astype(int)
        df['city_first_use'] = (df.groupby(['cc_num', 'city']).cumcount() == 0).astype(int)
        df['category_first_use'] = (df.groupby(['cc_num', 'category']).cumcount() == 0).astype(int)

        # Interaction features
        df['high_amt_night'] = ((df['amt'] > 100) & (df['is_night'] == 1)).astype(int)
        df['high_amt_far'] = ((df['amt'] > 100) & (df['distance_km'] > 100)).astype(int)
        df['new_merchant_high_amt'] = (df['merchant_first_use'] * (df['amt'] > 100)).astype(int)
        df['rapid_high_amt'] = ((df['is_rapid_succession'] == 1) & (df['amt'] > 100)).astype(int)
        df['weekend_night'] = (df['is_weekend'] & df['is_night']).astype(int)
        df['amt_distance_product'] = np.log1p(df['amt'] * df['distance_km'])

        # Amount unusual for this category
        df['category_amt_mean'] = df.groupby('category')['amt'].transform(
            lambda x: x.expanding().mean().shift(1).fillna(x.mean())
        )
        df['category_amt_std'] = df.groupby('category')['amt'].transform(
            lambda x: x.expanding().std().shift(1).fillna(x.std())
        )
        df['amt_zscore_category'] = (df['amt'] - df['category_amt_mean']) / (df['category_amt_std'] + 0.01)

        # Dormant card reactivation — silent card suddenly active with high amount
        df['is_dormant_reactivation'] = (
            (df['time_since_last_min'] > 10080) &  # >7 days gap
            (df['amt'] > df['amt_mean_card'])
        ).astype(int)

        df['new_category_high_amt'] = (
            (df['category_first_use'] == 1) & (df['amt'] > 100)
        ).astype(int)
        df['new_city_high_amt'] = (
            (df['city_first_use'] == 1) & (df['amt'] > 100)
        ).astype(int)

        # Add this in preprocess after pd.cut
        df['age_group'] = df['age_group'].astype(str)

        # Drop columns
        drop_cols = ['Unnamed: 0', 'first', 'last', 'street', 'trans_num', 'cc_num', 'category_amt_std',
                     'trans_date_trans_time', 'lat', 'long', 'merch_lat', 'merch_long', 'category_amt_mean',
                     'unix_time', 'dob', 'prev_merch_lat', 'prev_merch_long', 'prev_category']
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        # Reset index
        df = df.reset_index(drop=True)

        return df

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.preprocess(X)


def build_preprocess_pipe():
    return Pipeline([('preprocess', Preprocess())])
