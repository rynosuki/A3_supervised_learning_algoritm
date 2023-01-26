import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
import pickle

###
#
# The program seems to have some issues determining between label 0 and 6 with about a 10-15% failrate
# There is also some between 2,3,4,6
# Aswell as 0,2,4 for the label 6.
#
###


def main():
  trainset = np.genfromtxt("fashion-mnist_train.csv", delimiter=",")
  train_X, train_y = trainset[1:,1:], trainset[1:,0]
  testset = np.genfromtxt("fashion-mnist_test.csv", delimiter=",")
  test_X, test_y = testset[1:,1:], testset[1:,0]
  
  # choices = np.random.choice(len(train_X), 16, replace=False)
  # for i in range(16):
  #   plt.subplot(4,4,i+1)
  #   plt.imshow(train_X[choices[i]].reshape(28,28))
  #   plt.title(train_y[choices[i]])
  # plt.show()

  # Classifier has been pickled, and only loaded from now on!  
  # h = (500, 200, 56, 28, 14, 7)
  # clf = MLPClassifier(hidden_layer_sizes=h, alpha = 0.001, verbose = True, max_iter=400)
  # params = {'alpha': [0.001, 0.0001, 0.00001],'hidden_layer_sizes': [(500, 200, 56, 28, 14, 7), (500, 200, 56, 28, 14)]}
  # clf = GridSearchCV(mlp, params)
  # clf.fit(train_X, train_y)
  # with open("pickladneural.pkl", "wb") as file:
  #   pickle.dump(clf, file)
    
  with open("pickladneural.pkl", "rb") as file:
    clf = pickle.load(file)
    
  pred_y = clf.predict(test_X)
  matrix = np.zeros((10,10))
  for i in range(len(test_y)):
    matrix[int(pred_y[i]), int(test_y[i])] += 1
  plt.matshow(matrix)
  print(clf.score(train_X, train_y)*100)
  print(np.sum(pred_y == test_y)/100)
  for (i, j), z in np.ndenumerate(matrix):
    plt.text(j, i, '{:0.0f}'.format(z), ha='center', va='center')
  plt.show()
  
main()