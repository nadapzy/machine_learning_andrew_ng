# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 23:07:27 2015

@author: nadapzy
"""
def print_digits(images,y,max_n=20):
#    set up figure size in inches
    fig=plt.figure(figsize=(12,12))
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0.05,wspace=0.05)
    images=images.reshape([25,20,20])
    i=0
    while i<max_n and i<images.shape[0]:
#        plot the images in a mtrix of 20*20
        p=fig.add_subplot(20,20,i+1,xticks=[],yticks=[])
        p.imshow(ndimage.rotate(random.choice(images),90),cmap=plt.cm.bone,origin='lower')
        p.text(0,30,str(y[i]))
        i=i+1
        
def predict(theta1,theta2,x,k):
#    meh... don't know which activation function i am gonna use (sigmd is activation for log reg)
    hidden=sigmoid(x*theta1)
    hidden=np.insert(hidden,0,1,axis=1)
    output=sigmoid(hidden*theta2)
    pred=np.argmax(output,axis=1)   #+1
#    m=x.shape[0]
#    p=np.zeros(m,dtype='f')
#    for i in range(m):
#        xi=x[i,:]
#        hidden=sigmoid(xi*theta1)
#        hidden=np.insert(hidden,0,1,axis=1)
#        output=sigmoid(hidden*theta2)
#        p[i]=np.argmax(output)+1vc
    return pred

def max_ind(output):
    return np.argmax(output,axis=1)+1

def cost_func(theta,x,y,lam):
    theta1=theta[:10025].reshape(401,25)
    theta2=theta[10025:].reshape(26,10)
#    the activation function for neural network is quite different
#    this cost function actually shall cover the cost of all layers
    m=x.shape[0]
    hidden=sigmoid(x*theta1)
    hidden=np.insert(hidden,0,1,axis=1)
    '''Please do remember the vectorized cost function next time!!!!
    next step: regularization    
    '''
#    print('first part sum=',np.multiply(y,(-np.log(sigmoid(hidden*theta2)))))
    j=(1.0)/m*np.sum(np.multiply(y,(-np.log(sigmoid(hidden*theta2))))-np.multiply((1-y),np.log(1-sigmoid(hidden*theta2))))
    'I am changing the first row of theta1 and theta2 to 0 in order to calculate regularization term'
    theta1[0,:]=0
    theta2[0,:]=0    
    j+=lam/2/m*(np.sum(np.multiply(theta1,theta1))+np.sum(np.multiply(theta2,theta2)))
    return j

def sigmoidgrad(z):
    return np.multiply(sigmoid(z),(1-sigmoid(z)))
    
def sigmoid(z):
    return 1/(1+np.power(np.e,-z))
    
def accuracy(y,pred):
#     The following is for accuracy
#    pred=pred(theta,x)
    #Calculate accuracy:
    return (y==pred)

def randomtheta(l0,l1):
#    theta1=np.matrix(np.random.uniform(-0.12,0.12,size=l0*l1).reshape(l0,l1))
    theta=np.matrix(np.random.uniform(-0.12,0.12,size=l0*l1).reshape(l0,l1))
    return theta
  
def backprop(theta,x,y,lam):
    theta1=theta[:10025].reshape(401,25)
    theta2=theta[10025:].reshape(26,10)
    m=x.shape[0]
    hidden=sigmoid(x*theta1)
    hidden=np.insert(hidden,0,1,axis=1)
    output=sigmoid(hidden*theta2) #finish forward prop

    error3=output-y  #this is the error for layer 3
    delta3=(hidden.T)*error3
    theta2R=copy.copy(theta2)
    theta2R[0,:]=0
    delta3=delta3+lam*theta2R
#    there are some problems with the error2...! note!
#    try to understand the row-iterating formula!
#    error2=theta2.T*error3*sigmoidgrad(x*theta1)
    error2=np.multiply((error3*theta2.T),sigmoidgrad(np.insert(x*theta1,0,1,axis=1)))
    delta2=((x.T)*error2)[:,1:]
    theta1R=copy.copy(theta1)
    theta1R[0,:]=0
    delta2=delta2+lam*theta1R
    delta2=delta2/m
    delta3=delta3/m
    delta2=np.array(delta2).reshape(10025)
    delta3=np.array(delta3).reshape(260)
    delta=np.append(delta2,delta3)
    return delta
    
def grad_check(theta,x,y,lam):
#    theta1=theta[:10025].reshape(401,25)
#    theta2=theta[10025:].reshape(26,10)
    m=x.shape[0]
    epsilon=0.0001
    grad=[]
    for i in range(10285):
        print(i)
        thetai=np.zeros(10285)
        thetai[i]=epsilon
#        test=theta+thetai
        grad.append((cost_func(theta+thetai,x,y,lam)-cost_func(theta-thetai,x,y,lam))/(2*epsilon))
    return grad

def floatequal(delta,grad):
    delta1=copy.copy(delta)
    grad1=copy.copy(grad)
    delta1=(delta1*10**8).astype(int)
    grad1=(grad1*10**8).astype(int)
    return np.logical_not(delta1==grad1)
 
import scipy.io
from scipy import optimize,ndimage
import numpy as np
import matplotlib.pylab as plt
#import sklearn as sk
import matplotlib
#from sklearn.preprocessing import scale
import random
import copy

mat=scipy.io.loadmat('/users/nadapzy/documents/coding/mlclass-ex4-006/mlclass-ex4/ex4data1.mat')
thetas=scipy.io.loadmat('/users/nadapzy/documents/coding/mlclass-ex4-006/mlclass-ex4/ex4weights.mat')
Y=mat['y']
x=mat['X']
#print_digits(x,y)
theta1=np.mat(thetas['Theta1']).T  #theta1 is the hidden layer parameters(weights), in a shpae of (25,401)
theta2=np.mat(thetas['Theta2']).T  #theta2 is the output layer parameters, in a shpae of (10,401)
k=10
#x=x.reshape([5000,20,20])
#np.place(Y,Y==10,[0]) #replace number 10 in the data as number 0 
#print_digits(x,y)
#predict(theta1,theta2,x,k)
Y=Y-1
lam=1
x=np.mat(np.insert(x,0,1,axis=1))  # add 1's as the first column for matrix x
y=np.zeros((Y.shape[0],10))    # trying to convert 1,2,3,4 to K dimension arrays
for i in range(y.shape[0]):
    y[i,Y[i]]=1

l0=x.shape[1]
l1=26
l2=k
#randomize theta1 and theta2
theta1=np.array(randomtheta(l0,l1-1))
theta2=np.array(randomtheta(l1,l2))


theta1=np.array(theta1).reshape(10025)
theta2=np.array(theta2).reshape(260)
theta=np.append(theta1,theta2)
delta=backprop(theta,x,y,lam)
#grad=np.array(grad_check(theta,x,y,lam))

args=(x,y,lam)
resutl=optimize.fmin_cg(cost_func,theta,fprime=backprop,args=args,full_output=1,maxiter=400,disp=1,retall=1)
#,fprime=backprop

#cost=cost_func(theta1,theta2,x,y,lam)
#predict=predict(theta1,theta2,x,k)
#print('cost=',cost)
#pred=predict(theta1,theta2,x,k)
#np.place(pred,pred==10,[0])
#y=y.reshape(len(pred))
#accuracy=accuracy(y,pred)
#print('The accuracy rate=',np.sum(accuracy)/accuracy.shape[0])


