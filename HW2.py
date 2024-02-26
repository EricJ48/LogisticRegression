import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix
'''
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

mnist = load_digits()
x = mnist.data
y = mnist.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=None)
x_train = (x_train - np.mean(x_train)) / np.std(x_train)
x_test = (x_test - np.mean(x_test)) / np.std(x_test)
y_train = y_train.reshape(y_train.shape[0],1)
y_test = y_test.reshape(y_test.shape[0],1)
'''
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data() # convert data type to float 32
x_train=np.float32(x_train)

x_test=np.float32(x_test)

x_train = x_train.reshape(np.shape(x_train)[0], 28*28)

x_test = x_test.reshape(np.shape(x_test)[0], 28*28)
x_train = (x_train - np.mean(x_train)) / np.std(x_train)
x_test = (x_test - np.mean(x_test)) / np.std(x_test)
y_train = y_train.reshape(y_train.shape[0],1)
y_test = y_test.reshape(y_test.shape[0],1)


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

y_train0=(y_train==0).astype(int)
y_train1=(y_train==1).astype(int)
y_train2=(y_train==2).astype(int)
y_train3=(y_train==3).astype(int)
y_train4=(y_train==4).astype(int)
y_train5=(y_train==5).astype(int)
y_train6=(y_train==6).astype(int)
y_train7=(y_train==7).astype(int)
y_train8=(y_train==8).astype(int)
y_train9=(y_train==9).astype(int)

y_test0=(y_test==0).astype(int)
y_test1=(y_test==1).astype(int)
y_test2=(y_test==2).astype(int)
y_test3=(y_test==3).astype(int)
y_test4=(y_test==4).astype(int)
y_test5=(y_test==5).astype(int)
y_test6=(y_test==6).astype(int)
y_test7=(y_test==7).astype(int)
y_test8=(y_test==8).astype(int)
y_test9=(y_test==9).astype(int)

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

def forwardback(x,y,w,b):
  m = x.shape[0]
  dw = np.zeros((w.shape[0], 1))
  db = 0

  z = np.dot(x, w) + b
  sig = sigmoid(z)
  j = -(1 / m) * (np.dot(y.T, np.log(sig)) + np.dot((1 - y).T, np.log(1 - sig)))
  dw = (1 / m) * np.dot(x.T, (sig - y))
  db = (1 / m) * np.sum(sig - y)
  return j, dw, db

def predict(x, w, b):
  z = np.dot(x,w)+b
  Yhat_prob = sigmoid(z)
  Yhat = np.round(Yhat_prob).astype(int)
  return Yhat, Yhat_prob

def gradient_descent(x, y, w, b, alpha, iterations):
  costs = []
  for i in range(iterations):
    j, dw, db = forwardback(x, y, w, b)
    w = w - alpha * dw
    b = b - alpha * db
    costs.append(j)
  return costs, w, b

def LogisticRegression(x_train, x_test, y_train,y_test, alpha, iterations):
  features = x_train.shape[1]
  w = np.zeros((features,1))
  b = 0
  costs, w, b = gradient_descent(x_train, y_train, w, b, alpha, iterations)
  Predictions_train, _ = predict(x_train, w, b)
  Predictions, _ = predict(x_test, w, b)
  train_accuracy = accuracy_score(y_train, Predictions_train)
  test_accuracy = accuracy_score(y_test, Predictions)
  conf_matrix = confusion_matrix(y_test,Predictions)

  classifier = {"weights": w,
           "bias": b,
           "train_accuracy": train_accuracy,
           "test_accuracy": test_accuracy,
           "confusion_matrix": conf_matrix,
           "costs": costs}
  return classifier

classifiers_list = []
y_train_list = [y_train0, y_train1, y_train2, y_train3, y_train4, y_train5,
                y_train6, y_train7, y_train8, y_train9]
y_test_list = [y_test0, y_test1, y_test2, y_test3, y_test4, y_test5,
                y_test6, y_test7, y_test8, y_test9]
for i in range(10):
  logreg = LogisticRegression(x_train, x_test, y_train_list[i], y_test_list[i], 0.01, 100)
  print('Classifier', i, 'Accuracy:', logreg['test_accuracy'])
  classifiers_list.append(logreg)

def one_vs_all(data, classifiers_list):
  pred_matrix = np.zeros((data.shape[0], 10))
  for i in range(len(classifiers_list)):
    w = classifiers_list[i]['weights']
    b = classifiers_list[i]['bias']
    Yhat, Yhat_prob = predict(data, w, b)
    pred_matrix[:, i] = Yhat_prob.T
  max = np.amax(pred_matrix, axis=1, keepdims=True)
  predmax = (pred_matrix == max).astype(int)
  labels = []
  for j in range(predmax.shape[0]):
    idx = np.where(predmax[j, :] == 1)
    labels.append(idx)
  labels = np.vstack(labels).flatten()
  return labels


pred_label = one_vs_all(x_test, classifiers_list)
conf_matrix = confusion_matrix(y_test, pred_label)
totalAccuracy = accuracy_score(y_test, pred_label)
print('confusion')
print(conf_matrix)
print('accuracy')
print('Total Accuracy', totalAccuracy)

