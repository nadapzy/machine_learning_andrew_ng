# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 00:59:56 2015

@author: nadapzy
"""


#the data set contains a training set of housing prices
#the first column is the size of the house, the second column is the number of bedrooms
#and the third column is the price of the house.

def feature_scale(data):
    #step1: Feature normalization; substract the mean from all rows....
    data=data-np.mean(data,axis=0)
#then divide the feature values by their respective standard deviations
    data=data/np.std(data,axis=0)
    return data

def cost_func(x,y,m,n,theta):
#    cost=0
    return float((x*theta-y).T*(x*theta-y)/m    )

#def opt(x,y,m,n,alpha):
#    for i in range(m):
#        if i==0:        
#            delta=float(x[i,]*theta-y[i])*x[i,].T
#        else:
#            delta+=float(x[i,]*theta-y[i])*x[i,].T
#    delta=delta/m
#    return theta-alpha*delta       

#def normal(x,y,m,n,theta):
    


import numpy as np
from matplotlib import pylab as plt

data=np.genfromtxt('/users/nadapzy/documents/coding/ml/ex1/ex1data2.txt',delimiter=',')

#m is the number of traning sample; n is # of features
m=data.shape[0]
n=data.shape[1]

#x is features, y is output
x=data[:,:n-1]
y=np.matrix(data[:,n-1]).T
x=np.matrix(np.c_[[1]*m,feature_scale(x)])

#initiaze theta
theta=np.matrix([0]*n).T
alpha=0.01
#cost_func(x,y,m,n)
cost=cost_func(x,y,m,n,theta)
print('cost1=',cost_func(x,y,m,n,theta))
#ls_cost=[]
#for i in range(2000):
#    theta=opt(x,y,m,n,alpha)
#    ls_cost.append(cost_func(x,y,m,n,theta))
#    print('cost2=',ls_cost[i])
#
#x=np.arange(len(ls_cost))
#plt.figure()
#plt.ylim(0,1e10)
#plt.plot(x,ls_cost,'g-')

theta=np.linalg.inv(x.T*x)*x.T*y
cost=cost_func(x,y,m,n,theta)
print('cost2=',cost_func(x,y,m,n,theta))



