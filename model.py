from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from components.data_transformation import transformation

X, y = transformation(
    "C:/Daya/mlprojects/mlcarproject/artifacts/data/train.csv")

model = RandomForestClassifier()
model.fit(X, y)
print(model.score(X, y))
