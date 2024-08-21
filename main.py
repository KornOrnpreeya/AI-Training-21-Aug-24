# create streamlit app to load iris dataset from seaborn
import streamlit as st
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

df = sns.load_dataset('iris')
df

x = df.iloc[:, :-1]
y = df['species']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


st.sidebar.title('Classifiers')
classifier = st.sidebar.selectbox('Select Classifier', ('KNN', 'SVM','DecisionTreeClassifier','MLPClassifier'))
if classifier =='KNN'
  knn = KNeighborsClassifier(n_neighbors=1)
  knn.fit(x_train, y_train)
  y_pred = knn.predict(x_test)
  acc = accuracy_score(y_test, y_pred)
  st.write(acc)
if classifier =='SVM'
  svm = SVC()
  svm.fit(x_train, y_train)
  y_pred = svm.predict(x_test)
  acc = accuracy_score(y_test, y_pred)
  st.write(acc)

if classifier =='DecisionTreeClassifier'
  dt = DecisionTreeClassifier()
  dt.fit(x_train, y_train)
  y_pred = dt.predict(x_test)
  acc = accuracy_score(y_test, y_pred)
  st.write(acc)

if classifier =='MLPClassifier'
  nn = MLPClassifier()
  nn.fit(x_train, y_train)
  y_pred = nn.predict(x_test)
  acc = accuracy_score(y_test, y_pred)
  st.write(acc)
