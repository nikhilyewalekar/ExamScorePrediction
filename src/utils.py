import pandas as pd
import numpy as np
import pickle
import os
import sklearn

class PredictExamScore():
    def __init__(self):
        pass

    def predict_score(self, user_input_data):
        self.data = user_input_data
        self.create_test_df()
        self.predict = self.model.predict(self.test_df)
        print("Predicted Score is :",self.predict[0] )
        return np.round(self.predict[0],4)

    def create_test_df(self):
        self.load_model()
        test_array = np.zeros((1,self.model.input_shape[1]))

        test_array[0,0] = self.data["age"]
        test_array[0,1] = self.data["study_hours"]
        test_array[0,2] = self.data["class_attendance"]
        test_array[0,3] = self.data["sleep_hours"]

        feature = self.model.feature_names_in_

        gender = f'gender_{self.data["gender"]}'
        gender_index = np.where(feature == gender)[0]
        print("gender_index", gender_index)
        test_array[0,gender_index] = 1
 
        course = f'course_{self.data["course"]}'
        course_index = np.where(feature == course)[0]
        test_array[0,course_index] = 1

        internet_access = f'internet_access_{self.data["internet_access"]}'
        internet_access_index = np.where(feature == internet_access)[0]
        test_array[0,internet_access_index] = 1

        sleep_quality = f'sleep_quality_{self.data["sleep_quality"]}'
        sleep_quality_index = np.where(feature == sleep_quality)[0]
        test_array[0,sleep_quality_index] = 1

        study_method = f'study_method_{self.data["study_method"]}'
        study_method_index = np.where(feature == study_method)[0]
        test_array[0,study_method_index] = 1

        facility_rating = f'facility_rating_{self.data["facility_rating"]}'
        facility_rating_index = np.where(feature == facility_rating)[0]
        test_array[0,facility_rating_index] = 1

        exam_difficulty = f'exam_difficulty_{self.data["exam_difficulty"]}'
        exam_difficulty_index = np.where(feature == exam_difficulty)[0]
        test_array[0,exam_difficulty_index] = 1

        print("test_array", test_array)
        self.test_df = pd.DataFrame(test_array, columns = feature)

    def load_model(self):
        modelType = self.data["model_type"]
        print("modelType :",modelType)
        if modelType == "Linear":
            filepath = os.path.join("artifacts", "Linear_reg_model.pkl")
            with open(filepath, "rb") as f:
                self.model = pickle.load(f)
        elif modelType == "Descision":
            filepath = os.path.join("artifacts", "Descision_Tree_reg_model.pkl")
            with open(filepath, "rb") as f:
                self.model = pickle.load(f)
        elif modelType == "Knn":
            filepath = os.path.join("artifacts", "knn_reg_model.pkl")
            with open(filepath, "rb") as f:
                self.model = pickle.load(f)

