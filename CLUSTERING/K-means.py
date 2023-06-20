import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing, model_selection
import pandas as pd

style.use("ggplot")

# INITIAL VISUALISATION OF KMEANS ON TEST DATA

# X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])

# # plt.scatter(X[:, 0], X[:, 1], s=100)
# # plt.show()


# clf = KMeans(n_clusters=7)
# clf.fit(X)

# centroids = clf.cluster_centers_
# labels = clf.labels_
# # print(centroids, labels)
# colors = ["g.", "r.", "c.", "b.", "k.", "y."]

# for i in range(len(X)):
#     plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=10)

# plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=80)
# plt.show()


df = pd.read_excel("CLUSTERING/titanic.xls")

df.drop(["body", "name"], 1, inplace=True)


df.apply(pd.to_numeric, errors="ignore")
df.fillna(0, inplace=True)
# print(df.head())


def handle_non_num_data(df):
    columns = df.columns.values
    for col in columns:
        text_digit_vals = {}

        def convert_to_int_val(val):
            return text_digit_vals[val]

        if df[col].dtype != np.int64 and df[col].dtype != np.float64:
            column_contents = df[col].values.tolist()
            unique_el = set(column_contents)

            x = 0
            for unique in unique_el:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1
            df[col] = list(map(convert_to_int_val, df[col]))

    return df


df = handle_non_num_data(df)
# print(df.head())
df.drop(["boat", "sex"], 1, inplace=True)

X = np.array(df.drop(["survived"], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df["survived"])

clf = KMeans(n_clusters=2)
clf.fit(X)

correct = 0

for i in range(len(X)):
    predict_ = np.array(X[i].astype(float))
    predict_ = predict_.reshape(-1, len(predict_))

    prediction = clf.predict(predict_)

    if prediction[0] == y[i]:
        correct += 1

print(correct / len(X))
