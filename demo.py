import pandas as pd
from sklearn.model_selection import train_test_split

from classifiers.knn import KNNClassifier
from classifiers.pnn import PNNClassifier

df = pd.read_excel("./data/iris.xlsx")
X = df.drop(columns=["CLASS"])
y = df["CLASS"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=12)

knn = KNNClassifier(7)
knn.fit(X_train.values, y_train.values)

pnn = PNNClassifier()
pnn.fit(X_train.values, y_train.values)

knn_accuracy, knn_predictions = knn.detail_score(X_test.values, y_test.values)
pnn_accuracy, pnn_predictions = pnn.detail_score(X_test.values, y_test.values)

print(f"knn accuracy: {knn_accuracy}")
print(f"pnn accuracy: {pnn_accuracy}")
