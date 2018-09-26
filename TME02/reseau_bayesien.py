import random
import numpy as np
import matplotlib.pyplot as plt
import math
from tuto.fonctionHistogramme import *

def assertProba(p):
    assert 0<=p<=1

def bernoulli(p):
    return random.random()<p

def binomiale(n,p):
    return sum([bernoulli(p) for _ in range(n)])

def galton():
    """planche de Galton"""
    nbEtage = 10
    arr = np.array([binomiale(nbEtage,.5) for _ in range(100)])
    nbVals = nbEtage + 1
    a,b,c = plt.hist(arr,bins=range(nbVals))
    #plt.show()
    plt.close('all')

def normale(k, std):
    if k % 2 == 0:
        raise ValueError('le nombre k doit etre impair')

    N_0_STD = lambda x:math.exp(-x**2/2)/math.sqrt(2*math.pi)
    x = np.linspace(-2*std, 2*std, k)
    y = np.array([N_0_STD(i) for i in x])
    assert areProba(y)
    return x,y

def proba_affine(k, slope):
    if k % 2 == 0:
        raise ValueError('le nombre k doit etre impair')
    if abs(slope) > 2/(k*k):
        raise ValueError('pente trop raide, pente max = ' + str(2/(k*k)))

    cst = 1/k - (k-1)*slope/2
    x = np.linspace(0,1,k)
    y = np.array([cst + i*slope for i in x])
    assert areProba(y)
    return x,y

def pXY(x, y):
    """retourne le tableau p(x,y) des probas jointes, suppose indépendance
    x horizontal, y vertical"""
    res = np.array([y * valx for valx in x])
    assert areProba(res)
    return res

from mpl_toolkits.mplot3d import Axes3D

def dessine(P_jointe):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.linspace ( -3, 3, P_jointe.shape[0] )
    y = np.linspace ( -3, 3, P_jointe.shape[1] )
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, P_jointe, rstride=1, cstride=1 )
    ax.set_xlabel('A')
    ax.set_ylabel('B')
    ax.set_zlabel('P(A) * P(B)')
    plt.show ()

def independances():
    """Visualisation d'indépendances"""
    normx,normy = normale(1001,2)
    plt.plot(normx,normy)
    #plt.show()
    plt.close('all')

    affx, affy = proba_affine(101,.0001)
    plt.plot(affx, affy)
    plt.show()
    plt.close('all')

    norm_aff = pXY(normy, affy)
    dessine(norm_aff)

def main():
    galton()
    independances()

main()