import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model, preprocessing
import pickle

data_frame = pd.read_csv("shelfreachingcapacity.csv")

le = preprocessing.LabelEncoder()

#creating numerical data from the columns that had text data.
height = np.array(data_frame["heightin"])
stool_location_knowledge = le.fit_transform(list(data_frame["stoollocationknowledge"]))
can_reach_top_shelf = le.fit_transform(list(data_frame["canreachtopshelf"]))

X = list(zip(height, stool_location_knowledge))
y = list(can_reach_top_shelf)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

logistic_regression_object = linear_model.LogisticRegression()

model_accuracy = -1
best_accuracy = 0

for i in range(10000):
    logistic_regression_object.fit(x_train, y_train)
    model_accuracy = logistic_regression_object.score(x_test, y_test)
    if model_accuracy > best_accuracy:
        best_accuracy = model_accuracy
        #Makes sure that we can store the best model
        with open("predictionmodel.pickle", "wb") as f:
            pickle.dump(logistic_regression_object, f)

pickle_input = open("predictionmodel.pickle", "rb")
logistic_regression_object = pickle.load(pickle_input)

print("Coefficients of regression are: ", logistic_regression_object.coef_)
print("Accuracy was: ", best_accuracy)

prediction_vector = logistic_regression_object.predict(x_test)

for i in prediction_vector:
    print("Input value: ", x_test[i], "Predicted value: ", prediction_vector[i], "Actual value: ", y_test[i])

#More data would make the model better.