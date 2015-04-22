# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 14:26:13 2015

@author: nadapzy
"""
def gaussiankernel(x1,x2,sigma):
    k=np.exp(-(np.linalg.norm(x1-x2)**2)/2/sigma**2)
    return k
import scipy.io
from scipy import optimize,ndimage
import numpy as np
import matplotlib.pylab as plt
#import sklearn as sk
import matplotlib
#from sklearn.preprocessing import scale
import random
import copy

mat=scipy.io.loadmat('/users/nadapzy/documents/coding/mlclass-ex6-006/mlclass-ex6/ex6data1.mat')
#thetas=scipy.io.loadmat('/users/nadapzy/documents/coding/mlclass-ex4-006/mlclass-ex4/ex4weights.mat')
y=np.mat(mat['y'])
x=np.mat(mat['X'])
ysub=np.array(y).reshape(y.shape[0])  #have to transfer y to 1-d array in order to do subsetting
'try to plot ex6data1 out:'
#x1=x[ysub==1]
#x0=x[ysub==0]
#plt.plot(x1[:,0],x1[:,1],'b+')
#plt.plot(x0[:,0],x0[:,1],'yo')


