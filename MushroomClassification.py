import os
os.environ["PATH"] += os.pathsep + 'D:/Graphviz2.38/bin/'
import pandas as pd
import numpy as np
import math
import graphviz
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split as tts
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.tree import export_graphviz as exg
from sklearn import tree
from sklearn import preprocessing
from sklearn.metrics import classification_report

def main():

    df = pd.read_csv("mushrooms.csv")
    class_gillcolor = df[["class", "gill-color"]]

    #feat_importance(df)

    le = preprocessing.LabelEncoder()
    for column in df.columns:
        df[column] = le.fit_transform(df[column])

    y = df.iloc[:, 0]
    x = df.iloc[:, 1:21]

    """We can use train_test_split to split the dataframe into a testing and training set."""
    x_train, x_test, y_train, y_test = tts(x, y, test_size=0.1)

    """GaussianNB"""
    gaussian_nb_implementation(x_train, y_train, x_test, y_test)

    """Decision Tree"""
    decision_tree_implementation(df, x, x_train, y_train, x_test, y_test)

    """KNN"""
    k_nearest_neighbour(x_train, y_train, x_test, y_test)

    """plt.figure(figsize=(14, 12))
    sns.heatmap(df.corr(), linewidths=.1, cmap="YlGnBu", annot=True)
    plt.yticks(rotation=0);
    plt.show()"""


    """sns.set(style="ticks", color_codes=True)
    sns.barplot(x="class", y="habitat", data=df)
    plt.show()"""

    """Gill Colour Catplot"""
    """sns.catplot("class", col="gill-color", col_wrap=4, data=class_gillcolor, kind="count", height=2.5, aspect=.8)
    plt.show()"""


def gaussian_nb_implementation(x_train, y_train, x_test, y_test):
    print("GaussianNB")
    print("*************")

    clf = GaussianNB()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(classification_report(y_test, y_pred))


def decision_tree_implementation(df, x, x_train, y_train, x_test, y_test):
    print("Decision Tree")
    print("*************")

    my_tree = dtc(random_state=0)
    my_tree.fit(x_train, y_train)
    data = tree.export_graphviz(my_tree, out_file=None, feature_names=x.columns, filled=True, special_characters=True)
    graph = graphviz.Source(data)
    graph.render("mushroom")
    y_pred = my_tree.predict(x_test)
    print(classification_report(y_test, y_pred))


def k_nearest_neighbour(x_train, y_train, x_test, y_test):
    print("KNN")
    print("***")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    print(classification_report(y_test, y_pred))
    #a = list(knn.kneighbors_graph(y_pred))


def feat_importance(df):
    """Feature Importance"""
    le = preprocessing.LabelEncoder()
    for column in df.columns:
        df[column] = le.fit_transform(df[column])

    y = df.iloc[:, 0]
    x = df.iloc[:, 1:21]

    x_train, x_test, y_train, y_test = tts(x, y, test_size=0.1)

    feat_list = x.columns.values
    importances = dtc.feature_importances_
    indices = np.argsort(importances)

    plt.figure(figsize=(5, 7))
    plt.barh(y, range(len(indices)), importances[indices], align="center")
    plt.yticks(range(len(indices)), feat_list[indices])
    plt.xlabel('Importance')
    plt.title('Feature importances')
    plt.draw()
    plt.show()

main()
