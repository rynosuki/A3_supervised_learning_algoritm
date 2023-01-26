from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import lin_reg as lin

def main():
  data = np.genfromtxt("mnistsub.csv", delimiter=",")
  data1, data2, data3, data4, validationset = np.array_split(data, 5)
  data = np.concatenate((data1,data2,data3,data4))
  svc = SVC().fit(data[:,:2], data[:,2])
  
  figure, (ax1,ax2,ax3) = plt.subplots(1, 3)
  
  parameters = {'kernel':(['linear']), 'C':[1,10,100]}
  clf = GridSearchCV(svc, parameters)
  clf.fit(validationset[:,:2], validationset[:,2])
  print(clf.best_score_)
  
  temp_svc = SVC(C = clf.best_params_["C"], kernel = "linear")
  temp_svc.fit(data[:,:2], data[:,2])
  k = temp_svc.predict(data[:,:2])
  ax1.contour(dec_boundry(200, temp_svc, data[:,:2]).T, origin = "lower", extent=(min(data[:, 0])- 0.5, max(data[:, 0]) + 0.5,
                             min(data[:, 1])- 0.5, max(data[:, 1])+ 0.5))
  
  parameters = {'kernel':(['rbf']), 'C':[1,10,100], 'gamma':[2,4,6]}
  clf = GridSearchCV(svc, parameters)
  clf.fit(validationset[:,:2], validationset[:,2])
  print(clf.best_score_)
  
  temp_svc = SVC(C = clf.best_params_["C"], kernel = "rbf",gamma = clf.best_params_["gamma"])
  temp_svc.fit(data[:,:2], data[:,2])
  k = temp_svc.predict(data[:,:2])
  ax2.contour(dec_boundry(200, temp_svc, data[:,:2]).T, origin = "lower", extent=(min(data[:, 0])- 0.5, max(data[:, 0]) + 0.5,
                             min(data[:, 1])- 0.5, max(data[:, 1])+ 0.5))
  
  parameters = {'kernel':(['poly']), 'C':[1,10,100], 'degree':[2,3,4]}
  clf = GridSearchCV(svc, parameters)
  clf.fit(validationset[:,:2], validationset[:,2])
  print(clf.best_score_)
  
  temp_svc = SVC(C = clf.best_params_["C"], kernel = "poly", degree = clf.best_params_["degree"])
  temp_svc.fit(data[:,:2], data[:,2])
  k = temp_svc.predict(data[:,:2])
  ax3.contour(dec_boundry(200, temp_svc, data[:,:2]).T, origin = "lower", extent=(min(data[:, 0])- 0.5, max(data[:, 0]) + 0.5,
                             min(data[:, 1])- 0.5, max(data[:, 1])+ 0.5))
  
  ax1.scatter(data[:,0],data[:,1], c = data[:,2])
  ax2.scatter(data[:,0],data[:,1], c = data[:,2])
  ax3.scatter(data[:,0],data[:,1], c = data[:,2])
  plt.show()
  
def dec_boundry(grid_size, skl, X):
  
  min_value_x, max_value_x = min(X[:, 0]) - 0.5, max(X[:, 0]) + 0.5
  min_value_y, max_value_y = min(X[:, 1]) - 0.5, max(X[:, 1]) + 0.5

  x_axis = np.linspace(min_value_x, max_value_x, grid_size)
  y_axis = np.linspace(min_value_y, max_value_y, grid_size)
  
  grid = np.zeros(shape=(len(x_axis), len(y_axis)))
  for row in range(grid_size):
    for column in range(grid_size):
      z = np.column_stack((x_axis[row],y_axis[column]))
      grid[row, column] = skl.predict(z)
  return grid

  
main()