import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import fonctionHistogramme as fhist
import sys

plt.close('all')
#import pickle as pkl
#import pdb
#pdb.set_trace()

def output():
    """création concaténation affichage écriture fichier"""
    mat = np.empty((10, 3))
    for i in range(0, 10):
        mat[i][0] = i+1
        mat[i][1] = random.uniform()
        mat[i][2] = 0
    mat = np.vstack((range(1, 4), mat))#copie toute la matrice
    np.set_printoptions(precision=3)
    print(mat)
    np.savetxt("mat1", mat, fmt='%.5f', delimiter=',')#todo expliquer

def read(filename):
    return np.loadtxt(filename)

def gaussArray(n, moy, sig, vmin=-sys.maxsize, vmax=sys.maxsize):
    res = random.normal(moy, sig, n)
    #ou random.randn()*sig+moy
    res = np.round(res)
    res = np.maximum(res,vmin)
    res = np.minimum(res,vmax)
    return res

#partie1
output()
m = read('college.dat')
print(m.mean(0), m.std(0))#axe 0 colonnes 1 lignes

#partie2
n1 = gaussArray(15,10,4,0,20)
n2 = gaussArray(15, 8,4,0,20)
print("mean %.2f std %.2f" %(n1.mean(), n1.std()), n1)
print("mean %.2f std %.2f" %(n2.mean(), n2.std()), n2)

m2 = np.empty((5,5), int)
for i in range(5):
    for j in range(5):
        if (i+j)%2==0:
            m2[i][j] = np.random.randint(0,100)
        else:
            m2[i][j] = i+j
print(m2)
m3 = np.array([[i+j if (i+j)%2 else
                np.random.randint(0,100) for j in range(5)] for i in range(5)])
print(m3)

#3affichage
data = np.loadtxt("dataSalaire.csv", delimiter=';')
plt.scatter(range(8),data[:,1], c = np.array(['#00FF00']*len(data)), label='H|F')
plt.scatter(range(8),data[:,3], c = np.array(['#FFAAAA']*len(data)), label='F')
plt.scatter(range(8),data[:,5], c = np.array(['#0000FF']*len(data)), label='H')
plt.legend()
plt.xlabel('diplôme')
plt.ylabel('salaire horaire')
_, b = plt.xticks(range(8), ["Aucun", "BEPC", "CAP/BEP", "Bac", "IUT, BTS, DEUG",
                      ">Bac+2", "Master/Phd", "Ecole Ing/com"])
plt.setp(b, rotation=10, fontsize=8)

plt.figure()
plt.scatter(data[:,2],data[:,3], c = np.array(['#FFAAAA']*len(data)))
plt.plot(data[:,4],data[:,5], "ro")
plt.xlabel('temps travail')
plt.ylabel('salaire horaire')

#C = np.random.rand(5,10)
#plt.figure()
#plt.imshow(C, interpolation='nearest')# affichage sour forme d'image
#plt.colorbar()# legende

#tri selon y : a.sort(key=lambda x: x[1])
#index, = np.where(a[:,0]<0.5) #/!\ tuple constitué d'un seul nombre
#4Find
v = np.random.standard_normal(1000)
print("taille ", len(v), "mean %.3f std %.3f" %(v.mean(), v.std()))
print("diff <0 >0 / tot", abs(len(*np.where(v<0)) - len(*np.where(v>0)))/len(v))
print("-sig<x<sig / tot: ", 1 - len(*np.where((v<-1) | (v>1)))/len(v))

data = np.loadtxt("agePopulation.csv")
print("dim", len(data), len(data[0]), data[0])

plt.figure()
fhist.pltbarCompacte(data[:,4]/1000000,5)
plt.xlabel("tranche d'âge")
plt.ylabel('effectif (million)\ntotal: ' + str(np.sum(data[:,4])/1000000))
plt.savefig("tuto.pdf")
plt.show()

