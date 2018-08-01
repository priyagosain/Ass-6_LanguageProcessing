# This code is for plotting the Confusion matrix that pertains to Decision Trees

#referenced from http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html
#referenced from http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
# Importing the required packages

from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

#  Importing the training Dataset
filename = 'train.txt'

data = []
labels = []
with open(filename, encoding="utf-8") as f:
    for line in f.readlines():
        data.append(line.split("\t")[0])
        labels.append(line.split("\t")[1])

labels = np.array(labels)
class_names = ['bg', 'bs', 'cz', 'es-AR', 'es-ES', 'hr', 'id', 'mk', 'my', 'pt-BR', 'pt-PT', 'sk', 'sr', 'xx']

#to split the dataset
data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.3, random_state=100)
vectorizer = CountVectorizer()

# This dataset is way too high-dimensional. Better do PCA:
pca = TruncatedSVD(n_components=10)

# To select the K-best features if skipped before
selection = SelectKBest(chi2, k=11)

# Combining the above selected features through FeatureUnion
combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])

# Creating the classifier object
clf = DecisionTreeClassifier(criterion="gini", random_state=100)

# Performing training
pipeline = Pipeline([("count_vectorizer", vectorizer), ("features", combined_features), ("dt", clf)])
pipeline1 = pipeline.fit(data_train, labels_train)

# Function to make predictions
y_pred = pipeline1.predict(data_test)

# Function to plot the confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Compute confusion matrix
cnf_matrix = confusion_matrix(labels_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization (Decision Tree)')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix (Decision Tree)')

plt.show()

print("Accuracy of Decision Tree : ",
          accuracy_score(labels_test, y_pred) * 100)

