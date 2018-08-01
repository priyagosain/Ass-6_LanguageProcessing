
#This code is for plotting the training datasets
# random selection of thh data points has been made
# for visualization using the matlplot library
#TruncatedSVD is used to reduce the dimensions

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import numpy as np
import random

filename = 'train.txt'

data = []
labels = []
with open(filename,encoding="utf-8") as f:
    for line in f.readlines():
        data.append(line.split("\t")[0])
        labels.append(line.split("\t")[1])

labels = np.array(labels)
print("labels:", labels)
vectorizer = CountVectorizer()

X = vectorizer.fit_transform(data)

# print((X.shape))
svd = TruncatedSVD(n_components=10, n_iter=7, random_state=42)
svd = svd.fit(X)
X = svd.transform(X)
# print(len(X))


y = labels
print("labels1", labels)
target_names = ['bg', 'mk', 'bs', 'hr', 'sr', 'cz', 'sk', 'es_AR', 'es_ES', 'pt-BR', 'pt-PT', 'id', 'my', 'xx']
target_names = np.array(target_names)
print("traget_names", target_names)
# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(svd.explained_variance_ratio_))

fig, ax = plt.subplots()

colors = [
    'aliceblue',
    'antiquewhite',
    'aqua',
    'aquamarine',
    'azure',
    'beige',
    'bisque',
    'black',
    'blanchedalmond',
    'blue',
    'blueviolet',
    'brown',
    'burlywood']
for i in random.sample(range(len(X)),5005):

    if y[i] == 'bg\n':
        ax.scatter(X[i, 0], X[i, 1], color=colors[0], label='bg')
        continue

    if y[i] == 'mk\n':
        ax.scatter(X[i, 0], X[i, 1], color=colors[1], label='mk')
        continue

    if y[i] == 'bs\n':
        ax.scatter(X[i, 0], X[i, 1], color=colors[2], label='bs')
        continue

    if y[i] == 'hr\n':
        ax.scatter(X[i, 0], X[i, 1], color=colors[3], label='hr')
        continue

    if y[i] == 'sr\n':
        ax.scatter(X[i, 0], X[i, 1], color=colors[4], label='sg')
        continue

    if y[i] == 'cz\n':
        ax.scatter(X[i, 0], X[i, 1], color=colors[5], label='cz')
        continue

    if y[i] == 'sk\n':
        ax.scatter(X[i, 0], X[i, 1], color=colors[6], label='sk')
        continue

    if y[i] == 'es_AR\n':
        ax.scatter(X[i, 0], X[i, 1], color=colors[7], label='es_AR')
        continue

    if y[i] == 'es_ES\n':
        ax.scatter(X[i, 0], X[i, 1], color=colors[8], label='es_ES')
        continue

    if y[i] == 'pt-BR\n':
        ax.scatter(X[i, 0], X[i, 1], color=colors[9], label='pt-BR')
        continue

    if y[i] == 'pt-PT\n':
        ax.scatter(X[i, 0], X[i, 1], color=colors[10], label='pt-PT')
        continue

    if y[i] == 'id\n':
        ax.scatter(X[i, 0], X[i, 1], color=colors[11], label='id')
        continue

    if y[i] == 'my\n':
        ax.scatter(X[i, 0], X[i, 1], color=colors[12], label='my')

# plt.legend(loc='best', shadow=False, scatterpoints=1)
# plt.title('SVD of dataset')
ax.grid(False)
plt.show()
