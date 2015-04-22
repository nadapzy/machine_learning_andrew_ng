# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 21:30:15 2015

@author: nadapzy
"""
def print_digits(images,max_n=20):
#    set up figure size in inches
    fig=plt.figure(figsize=(20,20))
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0.05,wspace=0.05)
    images=images.reshape([images.shape[0],32,32])
    i=0
    while i<max_n and i<images.shape[0]:
#        plot the images in a mtrix of 20*20
        p=fig.add_subplot(20,20,i+1,xticks=[],yticks=[])
        p.imshow(ndimage.rotate(random.choice(images),90),cmap=plt.cm.bone,origin='lower')
        #p.text(0,30,str(y[i]))
        i=i+1
        
def feature_scale(x):
    mean=np.mean(x,axis=0)
    std=np.std(x,axis=0)
    x=(x-mean)/std
    return x,mean,std

def dim_reduction(x,k):
    m=x.shape[0]
    sigma=(1.0/m)*x.T*x
    u,s,v=np.linalg.svd(sigma)
    ured=u[:,:k]
    z=x*ured
    return z,u
    
import scipy.io
import scipy
from scipy import optimize,ndimage
import numpy as np
import matplotlib.pylab as plt
#import sklearn as sk
import matplotlib
#from sklearn.preprocessing import scale
import random
import copy
#
#mat=scipy.io.loadmat('/users/nadapzy/documents/coding/mlclass-ex7-006/mlclass-ex7/ex7faces.mat')
#x=mat['X']

x=x.reshape((16384,3))
x=x[10000:12000,:]
index=index[10000:12000]
x,mean,std=feature_scale(x)
#plt.plot(x[:,0],x[:,1],'go')
k=2
x=np.mat(x)
z,u=dim_reduction(x,k)

xrec=z*u[:,:k].T
xrec=np.array(xrec)
x=np.array(x)
z=np.array(z)
#print_digits(xrec,max_n=100)
#print_digits(x,max_n=100)
for i in range(16):
    c=cm.hot(i/16,1)
    idx=(index==i)
    plt.plot(z[idx,0],z[idx,1],color=c,marker='o',lw=0)
#plt.plot(xrec[:,0],xrec[:,1],'go')




