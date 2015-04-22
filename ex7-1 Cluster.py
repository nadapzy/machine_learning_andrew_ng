# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 14:49:06 2015

@author: nadapzy
"""


def findClosestCentroids(x,centroids):
    k=centroids.shape[0]
    distance=np.zeros((x.shape[0],k))
    for i in range(x.shape[0]):
        for j in range(k):
            distance[i,j]=np.linalg.norm(x[i,:]-centroids[j,:])
    idx=np.argmin(distance,axis=1)
    cost=np.sum(np.min(distance,axis=1))
    return idx,cost

def computeMeans(x,idx,k):
    centroids=np.zeros((k,x.shape[1]))
    for i in range(k):
        centroids[i,:]=np.mean(x[idx==i],axis=0)
    return centroids

def kmeansInitCentroids(x,k):
#    centroids=np.random.uniform(low=0,high=10,size=(k,2))
    randidx=random.sample(range(x.shape[0]),k)
    centroids=x[randidx,:]
    return centroids    
    
def kmeans(x,k):
    #initialize centroids
    centroids=kmeansInitCentroids(x,k)
#    cent_xhist=centroids[:,0]
#    cent_yhist=centroids[:,1]    
    cent_hist=centroids
    itera=10
    for i in range(itera):
        idx,cost=findClosestCentroids(x,centroids)
        print('cost=',cost)
        centroids=computeMeans(x,idx,k)
#        cent_xhist=np.append(cent_xhist,centroids[:,0],axis=1)
#        cent_yhist=np.append(cent_yhist,centroids[:,1],axis=1)
        cent_hist=np.append(cent_hist,centroids,axis=0)
        if np.isnan(np.sum(centroids)):
            print('i=',i)
            break
    idx,cost=findClosestCentroids(x,centroids)
    return centroids,idx,cost
    
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

#mat=scipy.io.loadmat('/users/nadapzy/documents/coding/mlclass-ex7-006/mlclass-ex7/ex7data2.mat')
#x=mat['X']
x=scipy.misc.imread('/users/nadapzy/documents/coding/mlclass-ex7-006/mlclass-ex7/bird_small.png')
x=x.reshape((16384,3))

#c=np.zeros(x.shape[0]) # c[i] =j that minimized distance(x[i]-mu[j]) where mu is the position of centroid j

k=16
ls_cent=[]
ls_index=[]
ls_cost=[]
x=x/255
for i in range(1):
    cent,index,cost=kmeans(x,k-1)
    ls_cent.append(cent)
    ls_index.append(index)
    ls_cost.append(cost)
argmin=np.nanargmin(ls_cost)    
cent=ls_cent[argmin]
index=ls_index[argmin]
x1=cent[index,:]
x1=x1.reshape((128,128,3))
x=x.reshape((128,128,3))
x=x*255
plt.imshow(x1)

#centroids=kmeansInitCentroids(x,k-1)
#idx=findClosestCentroids(x,centroids)
#centroids=computeMeans(x,idx,k-1)

'the folowing is for plotting'
'''
l0=np.arange(33)
l1=l0[0:32:3]
l2=l0[1:32:3]
l3=l0[2:32:3]
plt.plot(x[:,0],x[:,1],color=cm.hot(15/16,1))
plt.plot(cent[l1,0],cent[l1,1],'g-')
plt.plot(cent[l2,0],cent[l2,1],'r-')
plt.plot(cent[l3,0],cent[l3,1],'b-')
'''


