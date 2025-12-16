import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
import matplotlib.pyplot as plt

import pickle

df = pd.read_csv('./data/Exam_Score_Prediction.csv')
df.drop(columns = ['student_id'], inplace = True)
x= df.drop(columns=['exam_score'])
y = df.exam_score

encoded = pd.get_dummies(x, dtype=int)

#normal_scale = MinMaxScaler()
#normal_arr = normal_scale.fit_transform(encoded)
#x_scaled = pd.DataFrame(normal_arr, columns=encoded.columns)
#x_train, x_test, y_train, y_test = train_test_split(x_scaled,y, test_size=0.2, random_state=10)

#std_scale = StandardScaler()
#std_arr = std_scale.fit_transform(encoded)
#x_scaled = pd.DataFrame(std_arr, columns=encoded.columns)
#x_train, x_test, y_train, y_test = train_test_split(x_scaled,y, test_size=0.2, random_state=10)

x_train, x_test, y_train, y_test = train_test_split(encoded,y, test_size=0.2, random_state=10)

knn_reg_model = KNeighborsRegressor(n_neighbors=13, p=2)
knn_reg_model.fit(x_train, y_train)

'''
#Find best K and P value
#for noraml KNN model 
#P = 1, K = 13 , 71, 68
#P = 2, K = 13 , 71, 65

p = 2
k_values = range(1,25)

train_r2_score = []
test_r2_score = []

for k in k_values:
    knn_reg_model = KNeighborsRegressor(n_neighbors=k, p=p)
    knn_reg_model.fit(x_train, y_train)
    train_r2_score.append(knn_reg_model.score(x_train, y_train))
    test_r2_score.append(knn_reg_model.score(x_test, y_test))

plt.plot(k_values, train_r2_score)
plt.plot(k_values, test_r2_score)
plt.show()

'''

y_pred_train = knn_reg_model.predict(x_train)
mse = mean_squared_error(y_train, y_pred_train)
print("MSE :", mse)
print("RMSE :",np.sqrt(mse))
print("MAE : ",mean_absolute_error(y_train, y_pred_train))
print("R-Squared :", knn_reg_model.score(x_train, y_train))

y_pred_test = knn_reg_model.predict(x_test)
mse = mean_squared_error(y_test, y_pred_test)
print("MSE :", mse)
print("RMSE :",np.sqrt(mse))
print("MAE : ",mean_absolute_error(y_test, y_pred_test))
print("R-Squared :", knn_reg_model.score(x_test, y_test))


with open("knn_reg_model.pkl", 'wb') as f:
    pickle.dump(knn_reg_model, f)