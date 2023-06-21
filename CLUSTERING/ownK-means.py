import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing, model_selection
import pandas as pd

style.use("ggplot")

# INITIAL VISUALISATION OF KMEANS ON TEST DATA

X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])

# plt.scatter(X[:, 0], X[:, 1], s=100)
# plt.show()

colors = ["g", "r", "c", "b", "k", "y"]


class K_means:
    def __init__(self, k=2, tol=0.001, MAX_ITER=300):
        self.k = k
        self.tol = tol
        self.MAX_ITER = MAX_ITER

    def fit(self, data):
        self.centroids = {}
        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.MAX_ITER):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for featureset in data:
                distances = [
                    np.linalg.norm(featureset - self.centroids[centroid])
                    for centroid in self.centroids
                ]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(
                    self.classifications[classification], axis=0
                )

            optimised = True

            for c in self.centroids:
                original_centroids = prev_centroids[c]
                cur_centroids = self.centroids[c]

                if (
                    np.sum(
                        (cur_centroids - original_centroids)
                        / original_centroids
                        * 100.0
                    )
                    > self.tol
                ):
                    optimised = False
            if optimised:
                break

    def predict(self, data):
        distances = [
            np.linalg.norm(data - self.centroids[centroid])
            for centroid in self.centroids
        ]

        classification = distances.index(min(distances))

        return classification


clf = K_means()

clf.fit(X)

# SELF TEST DATA (With visualisation of centroids and clusters)
# for centroid in clf.centroids:
#     plt.scatter(
#         clf.centroids[centroid][0],
#         clf.centroids[centroid][1],
#         marker="o",
#         color="k",
#         s=100,
#     )

# for classification in clf.classifications:
#     color = colors[classification]

#     for featureset in clf.classifications[classification]:
#         plt.scatter(featureset[0], featureset[1], marker="x", c=color, s=100)


# unknows = np.array([[1, 3], [8, 9], [4, 3], [10, 10]])

# for u in unknows:
#     classification = clf.predict(u)
#     plt.scatter(u[0], u[1], marker="*", c=colors[classification], s=100)

# plt.show()

# TITANIC DATASET TEST

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

X = np.array(df.drop(["survived"], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df["survived"])

clf = K_means()
clf.fit(X)

correct = 0

for i in range(len(X)):
    predict_ = np.array(X[i].astype(float))
    predict_ = predict_.reshape(-1, len(predict_))

    prediction = clf.predict(predict_)

    if prediction == y[i]:
        correct += 1

print(correct / len(X))
