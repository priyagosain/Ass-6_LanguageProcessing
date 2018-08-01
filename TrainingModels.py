# This python code is for creating the classifier learning models:
# LinearSVC, Logistic Regression, MultinomialNB and Decision Trees
# for predicting which row of data belongs to which group.

#referenced from http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html

# Importing the required packages
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import NMF

# Importing the Dataset
filename = 'train.txt'

data = []
labels = []
with open(filename, encoding="utf-8") as f:
    for line in f.readlines():
        data.append(line.split("\t")[0])
        labels.append(line.split("\t")[1])

labels = np.array(labels)
class_names = ['bg', 'bs', 'cz', 'es-AR', 'es-ES', 'hr', 'id', 'mk', 'my', 'pt-BR', 'pt-PT', 'sk', 'sr', 'xx']

data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.3, random_state=100)
vectorizer = CountVectorizer()

# TruncatedSVD to select the principal components:
pca = TruncatedSVD(n_components=2)

#NMF for MultinomialNB as a PCA technique
pca1 = NMF(n_components=2)

# K-best features to be selected
selection = SelectKBest(chi2, k=1)

# To combine the features for LinearSVC, Decision Trees and Logistic Regression
combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])
# To combine the features for MultinomialNB
combined_features1 = FeatureUnion([("pca", pca1), ("univ_select", selection)])

clf = DecisionTreeClassifier(criterion="gini", random_state=100)

# Performing training by creating one pipeline per classifier according to the respective
#combined features and classification algorithm
pipeline_logreg = Pipeline([("count_vectorizer", vectorizer), ("features", combined_features), ("Logreg", LogisticRegression())])
pipeline_svc = Pipeline([("count_vectorizer", vectorizer), ("features", combined_features), ("svm", LinearSVC())])
pipeline_dt = Pipeline([("count_vectorizer", vectorizer), ("features", combined_features), ("dt", clf)])
pipeline_nb = Pipeline([("count_vectorizer", vectorizer), ("features", combined_features1), ("gnb", MultinomialNB())])

# Function to make predictions by logreg
pipeline_logreg = pipeline_logreg.fit(data_train, labels_train)
# Predicton on test data
y_pred_logreg = pipeline_logreg.predict(data_test)

# Function to make predictions by svc
pipeline_svc = pipeline_svc.fit(data_train, labels_train)
# Predicton on test data
y_pred_svc = pipeline_svc.predict(data_test)

# Function to make predictions by Decision Trees
pipeline_dt = pipeline_dt.fit(data_train, labels_train)
# Predicton on test data
y_pred_dt = pipeline_dt.predict(data_test)

# Function to make predictions by MultinomialNB
pipeline_nb = pipeline_nb.fit(data_train, labels_train)
# Predicton on test data
y_pred_nb = pipeline_nb.predict(data_test)

#Capturing the Accuracy score of each classifier
print("Accuracy of Logistic Regression : ",
          accuracy_score(labels_test, y_pred_logreg) * 100)

print("Accuracy of LinearSVC : ",
          accuracy_score(labels_test, y_pred_svc) * 100)

print("Accuracy of Decision Trees : ",
          accuracy_score(labels_test, y_pred_dt) * 100)

print("Accuracy of MultinomialNB : ",
          accuracy_score(labels_test, y_pred_nb) * 100)