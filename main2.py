# input library
import numpy as np
x = 4*np.random.rand(100)
y = np.sin(2*x+1) + 0.1*np.random.rand(100)

plt.scatter(x, y)

# create side bar for select the category to see the accuracy
st.sidebar.title('Classifier Selection')
classifier = st.sidebar.selectbox('Select Classifier', ('KNN', 'SVM', 'Decision Tree', 'Random Forest', 'Neural Network'))

# use k-nn to perform regression on x,y
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(x.reshape(-1, 1), y)

# predict x using knn
y_pred = knn.predict(x.reshape(-1, 1))

# scatter plot to compare y and y_pred in y-axis along with x in x-axis
plt.scatter(x, y)
plt.scatter(x, y_pred)

# use VSM to perform regression on x,y
from sklearn.svm import SVR
svm = SVR()
svm.fit(x.reshape(-1, 1), y)

# predict x using SVR
y_pred = svm.predict(x.reshape(-1, 1))

# scatter plot to compare y and y_pred in y-axis along with x in x-axis
plt.scatter(x, y)
plt.scatter(x, y_pred)

# use Decision Tree to perform regression on x,y
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(x.reshape(-1, 1), y)

# predict x using Decision Tree
y_pred = dt.predict(x.reshape(-1, 1))

# scatter plot to compare y and y_pred in y-axis along with x in x-axis
plt.scatter(x, y)
plt.scatter(x, y_pred)

# useRandom Forestto perform regression on x,y
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(x.reshape(-1, 1), y)

# predict x using Random Forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(x.reshape(-1, 1), y)

# scatter plot to compare y and y_pred in y-axis along with x in x-axis
plt.scatter(x, y)
plt.scatter(x, y_pred)

# use Neural Network to perform regression on x,y
from sklearn.neural_network import MLPRegressor
nn = MLPRegressor()
nn.fit(x.reshape(-1, 1), y)

# predict x using Neural Network
y_pred = nn.predict(x.reshape(-1, 1))

# scatter plot to compare y and y_pred in y-axis along with x in x-axis
plt.scatter(x, y)
plt.scatter(x, y_pred)

