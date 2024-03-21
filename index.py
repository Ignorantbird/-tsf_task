'''predict the percentage of a student bvased on the no. of study hours and also print what will be predicted score if the student studies for 9.2hrs/day'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

'''load the data '''
data = pd.read_csv("https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv")

'''split the data for feature extraction'''
study_hours = data.iloc[:, :-1]
percentage_score = data.iloc[:, 1]

'''split the data into training set and test set'''
study_hours_train, study_hours_test, percentage_score_train, percentage_score_test = train_test_split(study_hours, percentage_score, test_size=0.5, random_state=0)
#test_size is 0.2 which mean I am using 5
# 0% of the data for testing sets

rgrsr = LinearRegression()
'''training of linear regression model'''
rgrsr.fit(study_hours_train, percentage_score_train)

hrs = np.array([[9.2]])
result = rgrsr.predict(hrs.reshape(-1, 1))

print("If the student will study for 9.2 hours/day, he can score", result[0])

