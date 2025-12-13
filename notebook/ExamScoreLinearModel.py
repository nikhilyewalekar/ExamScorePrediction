import pandas as pd
import numpy as np

import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score

df = pd.read_csv("Exam_Score_Prediction.csv")
print(df.info())
x = df.drop(columns = ['exam_score'])
y = df.exam_score
encoded = pd.get_dummies(x, dtype = int)
encoded.drop(columns = ['student_id'], inplace = True)
x_train,x_test,y_train,y_test = train_test_split(encoded, y,test_size=0.2, random_state=10)
linear_model = LinearRegression()
linear_model.fit(x_train, y_train)

with open("Linear_reg_model.pkl", 'wb') as f:
    pickle.dump(linear_model, f)