import numpy as np
from scipy import rand
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def main(): 
  n_s = 5000
  X = np.genfromtxt("bm.csv", delimiter=",")
  y = X[:,2]
  X = X[:,:2]
  np.random.seed(13)
  r = np.random.permutation(len(y))
  X_s, y_s = X[:n_s, :], y[:n_s]
  X, y = X[r, :], y[r]
  
  svclassifier = SVC(kernel="rbf", C = 20, gamma=.5)
  svclassifier.fit(X,y)
  y_pred = svclassifier.predict(X)
  print(confusion_matrix(y, y_pred))
  print(classification_report(y, y_pred))
  print(svclassifier.support_)
  for i in svclassifier.support_:
    plt.plot(X[i, 0], X[i, 1], "ro")
  plt.scatter(X[:,0], X[:,1], c = y, marker=".")
  plt.show()
  
main()