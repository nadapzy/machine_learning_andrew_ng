# -*- coding: utf-8 -*-
"""
Spyder Editor

Vectorized implementation of linear regression!
Only for 1 input variables;
need more adjustments for multi-variable regression :)
"""

def cost_func(theta,m,x,y):
    cost=0
    for i in range(m):
        cost=cost+float((theta*x[i].transpose()-y[0,i])**2)
    return cost/2/m

def opt(theta,m,x,y,alpha):
    delta=0
    for i in range(m):
        if i==0:
            delta=float(theta*x[i].transpose()-y[0,i])*x[i].transpose()
        else:
            delta+=float(theta*x[i].transpose()-y[0,i])*x[i].transpose()
    delta=delta/m*alpha
    print('delta',delta)
    return theta-delta.transpose()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import math

data=np.genfromtxt('ex1data1.txt',delimiter=',')
datatx=data.transpose()
#plt.plot(data)

#--scatterplot the data with the line color as white
plt.plot(datatx[0],datatx[1],'wo')

#initialize theta
theta=np.matrix([0,0])

m=len(datatx[0])
x=np.matrix([[1]*len(datatx[0]),datatx[0]]).transpose()
y=np.matrix(datatx[1])
iterations=1500
alpha=0.01

print(cost_func(theta,m,x,y))

ls_cost=[]
for i in range(100):
    theta=opt(theta,m,x,y,alpha)
    ls_cost.append(cost_func(theta,m,x,y))
    print('new cost function',ls_cost[i])
    




