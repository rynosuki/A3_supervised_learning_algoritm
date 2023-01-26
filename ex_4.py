import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

###
#
# It wasn't to surprising as were going over so many values.
# The benefit i can see is that we get a very overall high % correct value.
# However there might be occasions where it goes lower though very small.
# Downsides i can see is that it requires alot of workload. Having to do alot of trees.
#
###

def main():
  data = np.genfromtxt("bm.csv", delimiter=",")
  trainingset, testset = np.split(data, 2)
  trainingset = np.column_stack((trainingset, np.ones(trainingset.shape)))
  
  trees = []
  
  for i in range(100):
    g = np.random.choice(range(trainingset.shape[0]), size=trainingset.shape[0], replace=True)
    tr = data[g]
    Xt, yt = tr[:,:tr.shape[1]-1], tr[:,tr.shape[1]-1]
    trees.append(DecisionTreeClassifier().fit(Xt,yt))
    
  result = forest_pred(trees, testset)
  print("Generalized error rate:", round((100 - (np.sum(result == testset[:,2])/5000)*100),3),"%")
  
  grids = []
  for tree in trees:
    grids.append(dec_boundry(100, tree, trainingset[:,:2]))
    
  for i in range(len(grids)-1):
    plt.subplot(10,10, i+1)
    plt.contour(grids[i], colors = ["orange"])
    
  plt.subplot(10,10,100)
  plt.contour(dec_boundry(100, trees, trainingset[:,:2]))
  plt.show()
  
    
def forest_pred(trees, X, plt = False):
  result = np.zeros(X.shape[0])
   
  for tree in trees:
    pred = tree.predict(X[:,:2])
    result += pred
    if not(plt):
      generr.append(np.sum(pred == X[:,2]))
   
  zeros = np.where(result < 50)
  ones = np.where(result >= 50)
  
  result[zeros] = 0
  result[ones] = 1
  
  return result

def dec_boundry(grid_size, skl, X):
  min_value_x, max_value_x = min(X[:, 0]) - 0.5, max(X[:, 0]) + 0.5
  min_value_y, max_value_y = min(X[:, 1]) - 0.5, max(X[:, 1]) + 0.5

  x_axis = np.linspace(min_value_x, max_value_x, grid_size)
  y_axis = np.linspace(min_value_y, max_value_y, grid_size)
  
  xx, yy = np.meshgrid(x_axis, y_axis)
  cells = np.stack([xx.ravel(), yy.ravel()], axis = 1)
  
  if(type(skl) == list):
    grid = forest_pred(skl, cells, True)
  else:
    grid = skl.predict(cells)
  return grid.reshape(grid_size,grid_size)
  
generr = []
main()
print("Average error rate:", round(100 - (np.average(generr)/5000)*100, 3),"%")