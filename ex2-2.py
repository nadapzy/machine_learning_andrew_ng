# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 19:32:04 2015
@author: nadapzy

This file covers feature mapping, regularization, logistics regression
And feature scaling
It uses gradient descent to solve logistics regression
"""


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

def cost_vet(theta,x,y,lam):
    if type(theta)!=type(y):
        theta=np.mat(theta).T
    m=x.shape[0]
    j=(1./m)*(-y.T*np.log(sigmd(x*theta))-(1-y.T)*np.log(1-sigmd(x*theta)))
    j+=lam/2*np.sum(np.power(theta[1:,:],2))/m
    return j[0,0]

def gd(theta,x,y,lam):
    if type(theta)!=type(y):
        theta=np.mat(theta).T
    m=x.shape[0]
    n=x.shape[1]
    h=sigmd(x*theta)
    delta=h-y
    grad=np.mat([1.]*n).T
    theta1=copy.copy(theta)
    theta1[0]=0
    for j in range(n):
#        print('shape of delta.T',delta.T.shape)
        xj=np.mat(x[:,j]).T
#        print('shape of xj',xj.shape)
        grad[j]=delta.T*(xj)/m
        if j!=0:
            grad[j]+=lam*theta1[j]/m
    return grad           

def opt_vet(theta,x,y,lam):
#    print(type(theta))
    if type(theta)!=type(y):
        theta=np.mat(theta).T
#    print('theta.shape=',theta.shape)        
    m=x.shape[0]
#    print('x.shape=',x.shape)
    h=sigmd(x*theta)
#    print('h.shape',h.shape)
    delta=h-y
    n=theta.shape[0]
    grad=np.array([1]*n)
#    grad=x.T*(delta)
    grad=((delta.T)*x).T
#    print('grad.shape=',grad.shape)
    theta1=copy.copy(theta)
    theta1[0]=0
    grad=grad/m+theta1*lam/m
#    print('theta1.shape=',theta1.shape)
    return (grad)
    
def sigmoid(z):
    return 1/(1+np.power(np.e,-z))
    
def pred(theta,x):    
    xtheta=sigmd(x*theta)
    f=lambda a: 1 if a>=0.5 else 0
    fv=np.vectorize(f)
    predt=fv(xtheta)
    return predt

def feature_map(x):
    x1=x[:,0]
    x2=x[:,1]    
    degree =6
    m=x1.shape[0]
    out=np.mat([1]*m,dtype='f').T
    for i in range(1,degree +1):
        for j in range(i+1):
            column=np.multiply(np.power(x1,(i-j)),np.power(x2,j))
            out=np.hstack((out,column))
    return out
    

import numpy as np
from matplotlib import pylab as plt
from scipy.optimize import fmin_bfgs
import copy

data=np.genfromtxt('/users/nadapzy/documents/coding/mlclass-ex2-006/mlclass-ex2/ex2data2.txt',delimiter=',')

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
#x,mean,std=feature_scale(x)
#x0=feature_map(x0)
x=feature_map(np.mat(x))
#update matrix shape
m=x.shape[0]
n=x.shape[1]
#x=np.matrix(np.c_[[1]*m,x])
#xb1=inv_feature_scale(np.array(x),mean,std)

#initiaze theta
theta=np.matrix([0]*n).T
#theta=np.matrix([ 0.01393706, -0.01530436,  0.00439422, -0.04697237, -0.01122051,
#       -0.03303874, -0.01595681, -0.00662116, -0.00763237, -0.01972467,
#       -0.0370302 , -0.00203605, -0.01205871, -0.00299302, -0.03549196,
#       -0.0180464 , -0.00403077, -0.00308006, -0.0054023 , -0.00428341,
#       -0.02750005, -0.02915839, -0.00093873, -0.00598416, -0.00033962,
#       -0.00680857, -0.00132846, -0.03522196]).T
#theta=np.matrix(np.random.random_sample(3)).T
alpha=10
lam=0.1
#cost_func(x,y,m,n)
#cost=cost_vet(theta,x,y,lam)
print('cost1=',cost_vet(theta,x,y,lam))
ls_cost=[]

for i in range(5000):
    theta=theta-alpha*opt_vet(theta,x,y,lam)
    ls_cost.append(cost_vet(theta,x,y,lam))
    print('cost2=',ls_cost[i])
#    print(theta)
    

#The following is for learning curve plotting
t=np.arange(len(ls_cost))
#plt.figure()
#plt.ylim(0,1e10)
#plt.plot(t,ls_cost,'g-')    
'''
 The following is for accuracy
pred=pred(theta,x)
#Calculate accuracy:
aVp=np.c_[pred,y]
accuracy_mat=np.sum(aVp,axis=1)
f=lambda a:1 if (a==0 or a==2) else 0
fv=np.vectorize(f)
accu_rate=np.sum(fv(accuracy_mat))/m

'''

#calculate the dicision boundary from theta and cutoff value
x1=np.linspace(-1,1.5,100)
x2=np.linspace(-1,1.5,100)
z=np.zeros(shape=(len(x1),len(x2)))
#x0=np.vstack((x1,x2))
#x0=np.mat(x0).T

for i in range(len(x1)):
    for j in range(len(x2)):
        x0=np.mat([x1[i],x2[j]])
        z[i,j]=(feature_map(x0).dot(theta))
plt.figure()        
levels=np.array([0.5])
C=plt.contour(x1,x2,z,levels=levels)        
#plt.axes([0.5])
plt.plot(subset0[:,0],subset0[:,1],'yo',subset1[:,0],subset1[:,1],'k+')
plt.clabel(C,inline=True,fontsize=10)      


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
myargs=(x,y,lam)
#for optimizing learning rate purpose, we use fmin_bfgs in scipy to opt
#ret=fmin_bfgs(cost_vet,theta,args=myargs,full_output=1,disp=True,retall=True)
#fprime=grad,
#theta=np.mat(ret[0]).T
