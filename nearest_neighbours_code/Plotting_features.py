import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, model_selection
import pandas as pd

df = pd.read_csv('heart_disease.csv')

x1 = np.array(df['chol'])
x2 = np.array(df['trestbps'])
x3 = np.array(df['fbs'])

y = np.array(df['target'])


plt.scatter(x1[y == 0], x2[y == 0], color='red')
plt.scatter(x1[y == 1], x2[y == 1], color='blue')
plt.show()