# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 00:06:12 2015

@author: nadapzy
"""
def feature_scale(data):
    #step1: Feature normalization; substract the mean from all rows....
    mean=np.mean(data,axis=0)    
    std=np.std(data,axis=0)
    data=data-mean
#then divide the feature values by their respective standard deviations
    data=data/std
    return data,mean,std

def cost_func(theta,x,y,lam):
    if type(theta)!=type(y):
        theta=np.mat(theta).T
#    cost=0
    m=x.shape[0]
    theta1=copy.copy(theta)
    theta1[0]=0
    cost=(1/2/m)*np.sum(np.square(x*theta-y))+lam/2/m*np.sum(np.square(theta1))
#    return float((x*theta-y).T*(x*theta-y)/m    )
    return cost

def opt(theta,x,y,lam):
    if type(theta)!=type(y):
        theta=np.mat(theta).T
    m=x.shape[0]
    theta1=copy.copy(theta)
    theta1[0]=0
    grad=x.T*((x*theta)-y)+lam*theta1
    return np.array(grad/m).reshape(theta.shape[0])
#    .reshape(theta.shape[0])
def train_LR(theta,x,y,lam):
    args=(x,y,lam)
    return optimize.fmin_cg(cost_func,theta,fprime=opt,args=args,full_output=True,disp=True,retall=True)
def polyfeature(x,m):
    x1=x[:,1:].copy()
    for i in range(2,m+1):
        x1=np.append(x1,np.power(x[:,1:],i),axis=1)
    x1=np.append(x[:,0],x1,axis=1)
    return x1
import scipy.io
from scipy import optimize,ndimage
import numpy as np
import matplotlib.pylab as plt
#import sklearn as sk
import matplotlib
#from sklearn.preprocessing import scale
import random
import copy

mat=scipy.io.loadmat('/users/nadapzy/documents/coding/mlclass-ex5-006/mlclass-ex5/ex5data1.mat')
thetas=scipy.io.loadmat('/users/nadapzy/documents/coding/mlclass-ex4-006/mlclass-ex4/ex4weights.mat')
y=np.mat(mat['y'])
x=np.mat(mat['X'])
xtest=np.mat(mat['Xtest'])
ytest=np.mat(mat['ytest'])
xval=np.mat(mat['Xval'])
yval=np.mat(mat['yval'])

'Here comes feature scaling'
x,mean,std=feature_scale(x)
xval=(xval-mean)/std
xtest=(xtest-mean)/std

m=x.shape[0]
mval=xval.shape[0]
mtest=xtest.shape[0]
n=x.shape[1]+1
#plt.plot(x,y,'rx')
#x=np.matrix(np.c_[[1]*m,feature_scale(x)])
x=np.matrix(np.c_[[1]*m,x])
p=8
xpoly=polyfeature(x,p)
xval=np.matrix(np.c_[[1]*mval,xval])
xvalpoly=polyfeature(xval,p)
xtest=np.matrix(np.c_[[1]*mtest,xtest])
xtestpoly=polyfeature(xtest,p)

#initialize theta
n=xpoly.shape[1]
theta=np.mat([1]*n).T
alpha=0.0001
lam=0.01
cost=cost_func(theta,xpoly,y,lam)
print('cost1=',cost_func(theta,xpoly,y,lam))
ls_cost=[]

#for i in range(10):
#    theta=theta-alpha*opt(theta,x,y,lam)
#    ls_cost.append(cost_func(theta,x,y,lam))
#    print('cost2=',ls_cost[i])

#xchr=np.arange(len(ls_cost))
#plt.figure()
#plt.ylim(0,1e10)
#plt.plot(xchr,ls_cost,'g-')

#args=(x,y,lam)
cost_train=[]
cost_val=[]
#cost_train_arr=np.arrary()
#result=optimize.fmin_cg(cost_func,theta,fprime=opt,args=args,full_output=True,disp=True,retall=True)
#lamrange=[0,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10]
theta=np.array(theta).reshape(n)
for j in range(500):
    cost_train=[]
    cost_val=[]
    for i in range(1,xpoly.shape[0]):    
        sample=np.random.choice(12,size=i,replace=False)
        result=train_LR(theta,xpoly[sample,:],y[sample,:],lam)
        cost_train.append(cost_func(result[0],xpoly[sample,:],y[sample,:],lam))
        cost_val.append(cost_func(result[0],xvalpoly,yval,lam))
    if j==0:
       cost_train_arr=np.array(cost_train)
       cost_val_arr=np.array(cost_val)
    else:
       cost_train_arr=np.vstack((cost_train_arr,np.array(cost_train)))
       cost_val_arr=np.vstack((cost_val_arr,np.array(cost_val)))
'''
plt.plot(lamrange,cost_train,'b--')
plt.plot(lamrange,cost_val,'g-') 
opt_lam=np.argmin(cost_val)
result_opt=train_LR(theta,xpoly,y,lamrange[opt_lam])

cost_test=cost_func(result_opt[0],xtestpoly,ytest,lamrange[opt_lam])
'''
cost_train=np.mean(cost_train_arr,axis=0)
cost_val=np.mean(cost_val_arr,axis=0)

'starting plotting learning curve...'
plt.figure(1)
#plt.subplot(1,2,1)  
plt.title('Learning curve Lambda=1')
xchart=range(1,12)
plt.plot(xchart,cost_train,'b-',lw=4)
plt.plot(xchart,cost_val,'r-')
#plt.plot(x[:,1],y,'rx')
#plt.plot(x[:,1],x*result[0].reshape((2,1)),'b-')
'starting plotting predicted vs. actual'
#plt.subplot(1,2,2)
'''
plt.title('Predicted vs. Actual Lambda=1')
theta=np.array(theta).reshape(n)
result=train_LR(theta,xpoly,y,lam)
xchart=x[:,1:]*std+mean
plt.plot(xchart,y,'rx')
lsp=np.linspace(-60,40,num=30)
lsppoly=np.mat((lsp-mean)/std).T
lsppoly=np.c_[[1]*lsp.shape[0],lsppoly]
lsppoly=polyfeature(lsppoly,p)
plt.plot(lsp,lsppoly*result[0].reshape((n,1)),'b--')
'''
