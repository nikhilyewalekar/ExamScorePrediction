import pandas as pd
import numpy as np

import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv('./data/Exam_Score_Prediction.csv')
df.drop(columns = ['student_id'], inplace = True)
x= df.drop(columns=['exam_score'])
y = df.exam_score

encoded = pd.get_dummies(x, dtype=int)
x_train, x_test, y_train, y_test = train_test_split(encoded,y, test_size=0.2, random_state=10)

#descision_tree_model = DecisionTreeRegressor(criterion='squared_error', max_depth=None,
#                min_samples_split=2,
#                min_samples_leaf=1)
#descision_tree_model.fit(x_train, y_train)

"""
descision_tree_model = DecisionTreeRegressor()
hyp = {"criterion" :['squared_error', 'absolute_error'],
        "max_depth" :range(4,9),
        "min_samples_split" : range(8,15),
        "min_samples_leaf" : range(5,12)}

print("Descision Tree hyper paramter tuning")
gscv_dt_model = GridSearchCV(descision_tree_model, hyp, n_jobs=-1, cv = 5, verbose=2)
gscv_dt_model.fit(x_train, y_train)

descision_tree_model = gscv_dt_model.best_estimator_
print("Descision Tree best parameter :", descision_tree_model)
"""

descision_tree_model = DecisionTreeRegressor(max_depth=8, min_samples_leaf=11, min_samples_split=11)
descision_tree_model.fit(x_train, y_train)

y_pred_train = descision_tree_model.predict(x_train)
mse = mean_squared_error(y_train, y_pred_train)
print("MSE :", mse)
print("RMSE :",np.sqrt(mse))
print("MAE : ",mean_absolute_error(y_train, y_pred_train))
print("R-Squared :", descision_tree_model.score(x_train, y_train))

y_pred_test = descision_tree_model.predict(x_test)
mse = mean_squared_error(y_test, y_pred_test)
print("MSE :", mse)
print("RMSE :",np.sqrt(mse))
print("MAE : ",mean_absolute_error(y_test, y_pred_test))
print("R-Squared :", descision_tree_model.score(x_test, y_test))

with open("Descision_Tree_reg_model.pkl", 'wb') as f:
    pickle.dump(descision_tree_model, f)