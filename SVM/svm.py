import numpy as np
from sklearn import model_selection, svm
import pandas as pd

# accuracies = []

df = pd.read_csv("KNN/breast-cancer-wisconsin.data.txt")

# CLEANING DATA
df.replace("?" , -99999 , inplace = True)
df.drop(["id"],1 , inplace = True)

X = np.array(df.drop(["class"],1))
Y = np.array(df["class"])


# SPLITTING TESTING AND TRAINING DATA
x_train , x_test , y_train , y_test = model_selection.train_test_split(X , Y ,  test_size = 0.2)

# CREATING KNN CLASSIFIER MODEL 
model = svm.SVC(kernel = "linear")

model.fit(x_train , y_train)

accuracy = model.score(x_test , y_test)*100
print(accuracy)
# accuracies.append(accuracy)

# print(sum(accuracies)/len(accuracies))

# PREDICTION
example_measures = np.array([[4,2,1,1,1,2,3,2,1] , [4,2,1,2,2,2,3,2,1]])
example_measures = example_measures.reshape(len(example_measures), -1)

prediction = model.predict(example_measures)

print(prediction)