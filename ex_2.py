import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

###
#
# The best classifier was found to be both. They were very similar in end result
# It mixes up 4s with 2s and 5s alot. Aswell as 9s as 5s and 7s
# params = {"C":[8,5,10,12], "gamma":[1e-7, 5e-7, 8e-7]}
# Was the params tried
#
###

def main():
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  x_train = x_train.reshape(60000,784)
  x_test = x_test.reshape(10000,784)
  x_train_small = np.split(x_train, 60)[0]
  y_train_small = np.split(y_train, 60)[0]
  x_test_small = np.split(x_test, 100)[0]
  y_test_small = np.split(y_test, 100)[0]
  
  fig, (ax1,ax2) = plt.subplots(1,2)

  svc = SVC()
  # This is how i found the C and gamma values for +95% accuracy.
  # params = {"C":[8,5,10,12], "gamma":[1e-7, 5e-7, 8e-7]}
  # gsc = GridSearchCV(svc, params)
  # gsc.fit(x_train_small, y_train_small)

  svc = SVC(C = 8, gamma = 5e-7)
  svc.fit(x_train_small, y_train_small)
  pred_y = svc.predict(x_test)
  
  matrix = np.zeros((10,10))
  for i in range(len(y_test)):
    matrix[pred_y[i], y_test[i]] += 1
  ax1.matshow(matrix)

  for (i, j), z in np.ndenumerate(matrix):
        ax1.text(j, i, '{:0.0f}'.format(z), ha='center', va='center')

  svcs = []
  values = np.empty((len(y_test),10))
  for i in range(10):
    temp_train_y = np.copy(y_train_small)
    temp_train_y = (temp_train_y == i).astype(int)
    temp_svc = SVC(probability=True, C = 4, gamma = 0.0000005, kernel="rbf").fit(x_train_small, temp_train_y)
    values[:,i] = temp_svc.predict_proba(x_test)[:,1]
  prediction = np.argmax(values, axis = 1)
  
  matrix = np.zeros((10,10))
  for i in range(len(y_test)):
    matrix[prediction[i], y_test[i]] += 1
  ax2.matshow(matrix)
  print(np.sum(prediction == y_test)/100)
  
  for (i, j), z in np.ndenumerate(matrix):
        ax2.text(j, i, '{:0.0f}'.format(z), ha='center', va='center')
  
  plt.show()
  
main()
