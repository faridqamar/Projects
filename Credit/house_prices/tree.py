import pandas as pd
import numpy as np
import pylab as pl

from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

svm_clf = SVC()
neighbors_clf = KNeighborsClassifier()
clfs = [
    ("svc", SVC()),
    ("KNN", KNeighborsClassifier())
    ]
for name, clf in clfs:
    clf.fit(df[iris.feature_names], df.species)
    print name, clf.predict(iris.data)
    print "*"*80
    
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(df[iris.feature_names], df.species)
clf.predict(df[iris.feature_names])
pd.crosstab(df.species, clf.predict(df[iris.feature_names]))
