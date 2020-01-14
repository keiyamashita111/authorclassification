import numpy as np
import pandas as pd
from sklearn import model_selection

df = pd.read_csv('heart_disease.csv')


class knn:
    def __init__(self, k=5):
        self.k = k

    def fit(self, x, y):
        self.x = x
        self.y = y

    def predict(self, query):
        distance = np.linalg.norm(self.x - query, axis=1)
        index = np.argpartition(distance, self.k)
        values = index[:self.k]
        unique, counts = np.unique(self.y[values], return_counts=True)
        prediction = unique[np.argmax(counts)]
        return prediction

    def score(self, x, y):
        print(len(x))
        points = 0
        for i in range(len(x)):
            ans = self.predict(x[i])
            if ans == y[i]:
                points += 1
        print(points)
        score = points/(len(x))
        return score


x = np.array(df.drop(['target'], 1))
y = np.array(df['target'])


x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2)

clf = knn()
clf.fit(x_train, y_train)

accuracy = clf.score(x_test, y_test)
print(accuracy)

example_measures = np.array([67,1,0,127,227,0,1,71,0,1,1,0,2])
example_measures = example_measures.reshape(1, -1)

prediction = clf.predict(example_measures)
print(prediction)











