import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
import warnings
warnings.filterwarnings('ignore')


def transformation(datapath: str):
    df = pd.read_csv(datapath)
    df.drop(['Date', 'Model'], axis=1, inplace=True)
    df.Engine = df.Engine.map(
        {"DoubleÂ\xa0Overhead Camshaft": "Double Camshaft", "Overhead Camshaft": "Camshaft"})
    X = df.drop("Company", axis=1)
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False,
                        dtype='int16').set_output(transform='pandas')
    X_encoded = ohe.fit_transform(
        X.drop(['Annual Income', 'Price ($)'], axis=1))
    X = pd.concat([X_encoded, X[['Annual Income', 'Price ($)']]], axis=1)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    label = LabelEncoder()
    y_label = label.fit_transform(df['Company'].values)
    return X_scaled, y_label
