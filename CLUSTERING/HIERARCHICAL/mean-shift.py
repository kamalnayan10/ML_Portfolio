import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import MeanShift
from sklearn import preprocessing
import pandas as pd

style.use("ggplot")

df = pd.read_excel("CLUSTERING/titanic.xls")

original_df = pd.DataFrame.copy(df)

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

clf = MeanShift()
clf.fit(X)

labels = clf.labels_
clusters = clf.cluster_centers_

original_df["cluster_group"] = np.nan

n_clusters_ = len(np.unique(labels))

for i in range(len(X)):
    original_df["cluster_group"].iloc[i] = labels[i]

survival_rates = {}

for i in range(n_clusters_):
    tempdf = original_df[(original_df["cluster_group"] == float(i))]

    survival_cluster = tempdf[(tempdf["survived"] == 1)]

    survival_rate = len(survival_cluster) / len(tempdf)

    survival_rates[i] = survival_rate

print(survival_rates)
print(original_df)
