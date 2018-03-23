# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


# DATA_PATH = "../breast-cancer-wisconsin.data"
DATA_PATH = "../wdbc.data"



column_names =[
    'Sample code number',
    'Clump Thickness',
    'Uniformity of Cell Size',
    'Uniformity of Cell Shape',
    'Marginal Adhesion',
    'Single Epithelial Cell Size',
    'Bare Nuclei',
    'Bland Chromatin',
    'Normal Nucleoli',
    'Mitoses',
    'Class'
]

data = pd.read_csv(DATA_PATH, names = column_names)
print(data.shape)

# delete "?"
data = data.replace(to_replace= "?", value = np.nan)
data = data.dropna(how = "any")

X_train, X_test, Y_train, Y_test = train_test_split(
    data[ column_names[1:10] ],
    data[ column_names[10] ],
    test_size = 0.25,
    random_state = 33
)

# vec = DictVectorizer(sparse=False)
# X_train = vec.fit_transform(X_train.to_dict(orient = ''))


ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)

dtc = DecisionTreeClassifier()
dtc.fit(X_train, Y_train)

Y_predict = dtc.predict(X_test)
print(dtc.score(X_test, Y_test))
print(classification_report(Y_predict, Y_test, target_names=['Benign','Malignant']))
