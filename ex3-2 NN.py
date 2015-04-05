# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 23:15:10 2015

@author: nadapzy
"""
def predict(theta1,theta2,x,k):
#    meh... don't know which activation function i am gonna use (sigmd is activation for log reg)
    hidden=sigmoid(x*theta1)
    hidden=np.insert(hidden,0,1,axis=1)
    output=sigmoid(hidden*theta2)
    pred=np.argmax(output,axis=1)+1
#    m=x.shape[0]
#    p=np.zeros(m,dtype='f')
#    for i in range(m):
#        xi=x[i,:]
#        hidden=sigmoid(xi*theta1)
#        hidden=np.insert(hidden,0,1,axis=1)
#        output=sigmoid(hidden*theta2)
#        p[i]=np.argmax(output)+1
    return pred
    
def sigmoid(z):
    return 1/(1+np.power(np.e,-z))
    
def accuracy(y,pred):
#     The following is for accuracy
#    pred=pred(theta,x)
    #Calculate accuracy:
    return (y==pred)
    
import scipy.io
from scipy import optimize,ndimage
import numpy as np
import matplotlib.pylab as plt
#import sklearn as sk
import matplotlib
#from sklearn.preprocessing import scale
import random
import copy

mat=scipy.io.loadmat('/users/nadapzy/documents/coding/mlclass-ex3-006/mlclass-ex3/ex3data1.mat')
thetas=scipy.io.loadmat('/users/nadapzy/documents/coding/mlclass-ex3-006/mlclass-ex3/ex3weights.mat')
y=mat['y']
x=mat['X']
theta1=np.mat(thetas['Theta1']).T  #theta1 is the hidden layer parameters(weights), in a shpae of (25,401)
theta2=np.mat(thetas['Theta2']).T  #theta2 is the output layer parameters, in a shpae of (10,401)
k=10
#x=x.reshape([5000,20,20])
np.place(y,y==10,[0]) #replace number 10 in the data as number 0 
#print_digits(x,y)
#predict(theta1,theta2,x,k)

x=np.mat(np.insert(x,0,1,axis=1))

pred=predict(theta1,theta2,x,k)
np.place(pred,pred==10,[0])
#y=y.reshape(len(pred))
accuracy=accuracy(y,pred)
print('The accuracy rate=',np.sum(accuracy)/accuracy.shape[0])

