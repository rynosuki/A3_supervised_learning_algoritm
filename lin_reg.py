from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from numba import njit

def cost_log(X, y, beta):
  return - (1/len(X)*(y.T.dot(np.log(sigmoid_log(np.dot(X, beta)))) + (1-y).T.dot(np.log(1-sigmoid_log(np.dot(X, beta))))))

def cost_lin(X, y, beta):
  return ((np.dot(X,beta)- y).T.dot(np.dot(X,beta) - y))/len(X)

def gradient_log(X, beta, y, alpha):
  return beta - (alpha / len(X))*(X.T).dot(sigmoid_log(np.dot(X, beta)) - y)

def gradient_lin(X, beta, y, alpha):
  return beta - np.dot(alpha,X.T).dot(np.dot(X,beta) - y)

def sigmoid_log(z):
  return 1/(1 + np.exp(-z))

def normalize_eq(X, x):
  Xn = []
  for i in range(X.shape[1]):
    mu = np.mean(x[:,i])
    sigma = np.std(x[:,i])
    Xn.append((X[:,i] - mu) / sigma)
  return np.array(Xn).T

def mse(pred_y, y):
  return ((pred_y - y)**2).mean()

@njit
def normal_eq(X, y):
  return np.linalg.inv(np.dot(X.T,X)).dot(X.T).dot(y)

def training_errors(X, beta, y):
  p = np.dot(X, beta).reshape(-1,1)
  p = sigmoid_log(p)
  pp = np.round(p)
  yy = y.reshape(-1,1)
  return np.sum(yy!=pp)

def mapFeature(X1,X2,D, Ones=True): # Pyton
  if Ones:
    one = np.ones([len(X1),1])
    Xe = np.c_[one,X1,X2] # Start with [1,X1,X2]
  else:
    Xe = np.c_[X1,X2] # Start with [1,X1,X2]
  for i in range(2,D+1):
    for j in range(0,i+1):
      Xnew = X1**(i-j)*X2**j # type (N)
      Xnew = Xnew.reshape(-1,1) # type (N,1) required by append
      Xe = np.append(Xe,Xnew,1) # axis = 1 ==> append column
  return Xe

def plot_grid(X1, X2, beta, y, poly, plot):
  min_x, max_x = min(X1), max(X1)
  min_y, max_y = min(X2), max(X2)
  grid_size = 200
  x_axis = np.linspace(min_x - 0.1, max_x + 0.1, grid_size)
  y_axis = np.linspace(min_y - 0.1, max_y + 0.1, grid_size)
  
  xx, yy = np.meshgrid(x_axis, y_axis)
  x1, x2 = xx.ravel(), yy.ravel()
  XXe = mapFeature(x1, x2, poly)
  
  p = sigmoid_log(np.dot(XXe, beta))
  classes = p > 0.5
  clz_mesh = classes.reshape(xx.shape)
  
  cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
  cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
  
  plot.pcolormesh(xx,yy,clz_mesh, cmap=cmap_light)
  plot.scatter(X1, X2,c=y, marker='.', cmap=cmap_bold)
  return plot

def plot_grid_sklearn(X1, X2, y, plot, lg, poly):
  min_x, max_x = min(X1), max(X1)
  min_y, max_y = min(X2), max(X2)
  grid_size = 200
  x_axis = np.linspace(min_x - 0.1, max_x + 0.1, grid_size)
  y_axis = np.linspace(min_y - 0.1, max_y + 0.1, grid_size)
  
  xx, yy = np.meshgrid(x_axis, y_axis)
  x1, x2 = xx.ravel(), yy.ravel()
  XXe = mapFeature(x1, x2, poly, Ones=False)

  p = lg.predict(XXe)
  print(p)
  classes = p > 0.5
  clz_mesh = classes.reshape(xx.shape)
  
  cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
  cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
  
  plot.pcolormesh(xx,yy,clz_mesh, cmap=cmap_light)
  plot.scatter(X1, X2,c=y, marker='.', cmap=cmap_bold)