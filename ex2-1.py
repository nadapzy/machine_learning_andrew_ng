# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 00:59:56 2015

@author: nadapzy

This file uses gradient descent to solve logistics regression without regularization

Fmin_BFGS is almost used but commented out for result testing

Adjustment on Mar. 28:
Completely vectorize cost function and gradient function
called: cost_vet, and opt_vet

"""


#the data set contains a training set of housing prices
#the first column is the size of the house, the second column is the number of bedrooms
#and the third column is the price of the house.

def feature_scale(data):
    #step1: Feature normalization; substract the mean from all rows....
    mean=np.mean(data,axis=0)
    data=data-np.mean(data,axis=0)
#then divide the feature values by their respective standard deviations
    std=np.std(data,axis=0)
    data=data/np.std(data,axis=0)
    return data,mean,std
    
def inv_feature_scale(data,mean,std):
    data[:,1:]=data[:,1:]*std
    data[:,1:]=data[:,1:]+mean
    
    return data

def cost_func(theta,x,y,m,n):
#    cost=0
#    theta0=np.matrix(theta).T
    for i in range(m):
        xtheta=sigmd(x[i,]*theta)
        if i==0:
            cost=-y[i]*np.log(xtheta)-(1-y[i])*np.log(1-xtheta)
        else:
            cost+=-y[i]*np.log(xtheta)-(1-y[i])*np.log(1-xtheta)            
    return float(cost/m)

def cost_vet(theta,x,y):
    if type(theta)!=type(y):
        theta=np.mat(theta).T
    m=x.shape[0]
    j=(1./m)*(-y.T*np.log(sigmd(x*theta))-(1-y.T)*np.log(1-sigmd(x*theta)))
    return j[0,0]

def opt_vet(theta,x,y):
    if type(theta)!=type(y):
        theta=np.mat(theta).T
    h=sigmd(x*theta)
    m=x.shape[0]
    delta=h-y
    n=theta.shape[0]
    grad=np.array([1]*n)
    grad=x.T*(delta)
    return grad/m
#    for j in range(n):
#        sumdelta=delta.T*x[:,j]
#        grad[j]=(1./m)*sumdelta
    
    
def opt(theta,x,y,m,n,alpha):
    delta=np.array([1]*n,dtype='float')
    for j in range(n):
        for i in range(m):
            xtheta=sigmd(x[i,]*theta)
            if i==0:        
                delta[j]=float((xtheta-y[i])*x[i,j])
            else:
                delta[j]=delta[j]+float((xtheta-y[i])*x[i,j])
#                delta[j]=delta[j]+0.01
        delta[j]=float(delta[j]/m)
#    theta=theta-np.matrix(delta).T        
    return theta-alpha*np.matrix(delta).T        

#def normal(x,y,m,n,theta):
    
def sigmoid(z):
    return 1/(1+np.power(np.e,-z))
    
def pred(theta,x):    
    xtheta=sigmd(x*theta)
    f=lambda a: 1 if a>=0.5 else 0
    fv=np.vectorize(f)
    predt=fv(xtheta)
    return predt

import numpy as np
from matplotlib import pylab as plt
from scipy.optimize import fmin_bfgs
import copy

data=np.genfromtxt('/users/nadapzy/documents/coding/mlclass-ex2-006/mlclass-ex2/ex2data1.txt',delimiter=',')

#transfer sigmoid function to be elementwise for array and matrix
sigmd=np.vectorize(sigmoid)

#m is the number of traning sample; n is # of features
m=data.shape[0]
n=data.shape[1]

subset0=data[data[:,2]==0]
subset1=data[data[:,2]==1]
#plt.plot(subset0[:,0],subset0[:,1],'yo',subset1[:,0],subset1[:,1],'k+')

#x is features, y is output
x=data[:,:n-1]
y=np.matrix(data[:,n-1]).T
xb=copy.copy(x)
x,mean,std=feature_scale(x)
x=np.matrix(np.c_[[1]*m,x])
#xb1=inv_feature_scale(np.array(x),mean,std)

#initiaze theta
theta=np.matrix([0]*n).T
#theta=np.matrix(np.random.random_sample(3)).T
alpha=1
#cost_func(x,y,m,n)
cost=cost_vet(theta,x,y)
print('cost1=',cost_vet(theta,x,y))
ls_cost=[]

for i in range(2000):
    theta=theta-alpha*opt_vet(theta,x,y)
    ls_cost.append(cost_vet(theta,x,y))
    print('cost2=',ls_cost[i])
#    print(theta)
t=np.arange(len(ls_cost))
plt.figure()
#plt.ylim(0,1e10)
plt.plot(t,ls_cost,'g-')

pred=pred(theta,x)
#Calculate accuracy:
aVp=np.c_[pred,y]
accuracy_mat=np.sum(aVp,axis=1)
f=lambda a:1 if (a==0 or a==2) else 0
fv=np.vectorize(f)
accu_rate=np.sum(fv(accuracy_mat))/m


#calculate the dicision boundary from theta and cutoff value
#x1=-(theta[0])/theta[1]
#x1=float(x1*std[0]+mean[0])
#y1=-theta[0]/theta[2]
#y1=float(y1*std[1]+mean[1])
#plt.figure()
#plt.plot(subset0[:,0],subset0[:,1],'yo',subset1[:,0],subset1[:,1],'k+'
#,[x1,0],[0,y1],'k')


#theta=np.linalg.inv(x.T*x)*x.T*y
#cost=cost_func(x,y,m,n,theta)
#print('cost2=',cost_func(x,y,m,n,theta))

#Using generic optimization alg for minizing cost...
#Actually you can input the gradient function for better performance!
myargs=(x,y)
#for optimizing learning rate purpose, we use fmin_bfgs in scipy to opt
ret=fmin_bfgs(cost_vet,theta,args=myargs,full_output=1,disp=True,retall=1)


