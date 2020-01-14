import numpy as np
from sklearn import neighbors, model_selection
import pandas as pd

df = pd.read_csv('heart_disease.csv')

x = np.array(df.drop(['target'], 1))
y = np.array(df['target'])

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(x_train, y_train)

accuracy = clf.score(x_test, y_test)
print(accuracy)

example_measures = np.array([67,1,0,127,227,0,1,71,0,1,1,0,2])
example_measures = example_measures.reshape(1, -1)

prediction = clf.predict(example_measures)
print(prediction)