# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 22:59:08 2015

@author: nadapzy
"""
def print_digits(images,y,max_n=20):
#    set up figure size in inches
    fig=plt.figure(figsize=(12,12))
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0.05,wspace=0.05)
    images=images.reshape([5000,20,20])
    i=0
    while i<max_n and i<images.shape[0]:
#        plot the images in a mtrix of 20*20
        p=fig.add_subplot(20,20,i+1,xticks=[],yticks=[])
        p.imshow(ndimage.rotate(random.choice(images),90),cmap=plt.cm.bone,origin='lower')
        p.text(0,30,str(y[i]))
        i=i+1

def opt_vet(theta,x,y,lam):
#    issue: the input is not matrix!!!!fix it tomorrow
#    print(type(theta))
    if type(theta)!=type(y):
        theta=np.mat(theta).T  #ensure theta is inputed as matrix
#    print('theta.shape=',theta.shape)        
    m=x.shape[0]
    h=sigmd(x*theta)
    error=h-y
    n=theta.shape[0]
    grad=np.array([1]*n)
#    print('size of error mat=',type(error),error.shape)
#    print('size of x mat=',type(x),x.shape)
#    print('size of theta',type(theta),theta.shape)
#    print('size of y',type(y),y.shape)
    grad=((error.T)*x).T
#    The following 3 lines for regularization
    theta1=copy.copy(theta)
    theta1[0]=0
    grad=grad/m+theta1*lam/m
    grad=np.array(grad).reshape(401,)
#    print('size of grad',type(grad),grad.shape)
    return (grad)

def sigmoid(z):
    return 1/(1+np.power(np.e,-z))
    
def sigmd(z):
    return np.vectorize(sigmoid)(z)    

def cost_vet(theta,x,y,lam):
    if type(theta)!=type(y):
        theta=np.mat(theta).T   #ensure theta is inputed as a matrix
    m=x.shape[0]
    j=(1./m)*(-y.T*np.log(sigmd(x*theta))-(1-y.T)*np.log(1-sigmd(x*theta)))
    j+=lam/2*np.sum(np.power(theta[1:,:],2))/m  #this line is for regularization
    return j[0,0]

def one_vs_all(x,y,k):
#    for i in range(k):
    m,n=x.shape
    lam=1
    theta=np.mat(np.zeros(n+1)).T
    all_theta=np.zeros(shape=(k,n+1),dtype='f')
    x=np.mat(np.insert(x,0,1,axis=1))
    y=np.mat(y)
    for c in range(k):
        result=optimize.fmin_cg(
            cost_vet,theta,fprime=opt_vet,args=(x,(y==c).astype(int),lam)
            ,full_output=1,disp=1,retall=1)  #passing in boolean wouldn't work??
        all_theta[c]=result[0]
#    print('shape of result',result.shape) 
    return all_theta
        
def predictOneVSALL(all_theta,x,k):
    '''
    x is the input from the dumb octave file, in (m*n)
    all_theta is the trained theta, vstack together all k regular theta's together
    the way to predict is to calculate each sigmd(x*theta[:,i]), assuming i is iterator
    then compare to get the maximum i (use argmax function against a list)
    so that we get the prediction...
    '''
    x=np.insert(x,0,1,axis=1)
    if all_theta.shape[1]!=k:
        all_theta=np.mat(all_theta).T
    print('all_theta shape',type(all_theta),all_theta.shape)
    pred_mat=x*all_theta
    pred=np.argmax(pred_mat,axis=1)
    return pred

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

y=mat['y']
x=mat['X']
k=10
#x=x.reshape([5000,20,20])
np.place(y,y==10,[0]) #replace number 10 in the data as number 0 
#print_digits(x,y)

#all_theta=one_vs_all(x,y,k)

pred=predictOneVSALL(all_theta,x,k)

accuracy=accuracy(y,pred)
#Yay, the result is 0.944799999, about 95% accuracy :)





