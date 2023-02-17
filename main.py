import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split,from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("iris_headers.csv")

x = df.iloc[:, :-1]
y = df.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.2,
                                                    shuffle=True,  # shuffle the data to avoid bias
                                                    random_state=0)
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

knn = KNeighborsClassifier(metric="euclidean", algorithm="kd_tree", n_neighbors=2)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

print(f'The accuracy of sklearn implementation is {accuracy_score(y_test, y_pred)}')
