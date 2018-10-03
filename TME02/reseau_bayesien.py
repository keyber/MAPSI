import random
import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
from mpl_toolkits.mplot3d import Axes3D
import pyAgrum as gum
import pyAgrum.lib.ipython as gnb

def assertDensity(a):
    if abs(np.sum(a)-1)>.001 or not areProba(a):
        print(a.shape, a.size)
        print(a)
        print(np.sum(a))
        print(max(a))
        assert False

def areProba(a):
    return not np.any((a<0)|(a>1))

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
    """variable continue, retourne quelques évalutations de la fonction de densité,
    ne somme pas a 1"""
    if k % 2 == 0:
        raise ValueError('le nombre k doit etre impair')

    N_0_STD = lambda x:math.exp(-x**2/2)/math.sqrt(2*math.pi)
    x = np.linspace(-2*std, 2*std, k)
    y = np.array([N_0_STD(i) for i in x])
    assert areProba(y)
    return x+2*std,y

def proba_affine(k, slope):
    """variable continue, retourne quelques évalutations de la fonction de densité,
    ne somme pas a 1"""
    if k % 2 == 0:
        raise ValueError('le nombre k doit etre impair')
    cst = 1/k - (k-1)*slope/2

    if cst<0 or cst+(k-1)*slope>1:
        raise ValueError('mauvaise pente' + slope)

    x = np.linspace(0,k-1,k)
    y = np.array([cst + i*slope for i in x])
    assert areProba(y)
    return x,y

def pXY(x, y):
    """retourne le tableau p(x,y) des probas jointes, suppose indépendance
    x horizontal, y vertical"""
    res = np.array([y * valx for valx in x])
    assert areProba(res)
    return res

def dessine(P_jointe):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x = np.linspace (0, 10, P_jointe.shape[1])
    y = np.linspace (0, 10, P_jointe.shape[0])
    X, Y = np.meshgrid(x, y)
    print(X.shape, Y.shape, P_jointe.shape)
    ax.plot_surface(X, Y, P_jointe)
    ax.set_xlabel('A')
    ax.set_ylabel('B')
    ax.set_zlabel('P(A) * P(B)')
    plt.show()

def visualisationIndependances():
    """Visualisation d'indépendances"""
    normx,normy = normale(55,2)
    plt.plot(normx,normy)
    #plt.show()
    plt.close('all')

    affx, affy = proba_affine(7,.04)
    plt.plot(affx, affy)
    #plt.show()
    plt.close('all')

    norm_aff = pXY(normy, affy)
    print(norm_aff)
    dessine(norm_aff)

def independancesConditionnelles():
    P_XYZT = np.array([[[[0.0192, 0.1728],
                         [0.0384, 0.0096]],

                        [[0.0768, 0.0512],
                         [0.016, 0.016]]],

                       [[[0.0144, 0.1296],
                         [0.0288, 0.0072]],

                        [[0.2016, 0.1344],
                         [0.042, 0.042]]]])
    assertDensity(P_XYZT)

    P_YZ = np.array([[np.sum(P_XYZT[:,y,z,:])
                     for z in range(2)] for y in range(2)])
    assertDensity(P_YZ)

    #x à l'intérieur
    P_XT_YZ = np.array([P_XYZT[x,y,z,t] / P_YZ[y,z] for x,t,y,z in
                        itertools.product(*map(range, [2,2,2,2]))]).reshape(2,2,2,2)

    #x extérieur
    P_X_YZ = np.array([[[np.sum(P_XT_YZ[x,:,y,z]) for z in range(2)]
                       for y in range(2)] for x in range(2)])

    #intérieur
    P_T_YZ = np.array([np.sum(P_XT_YZ[:,t,y,z]) for t in range(2)
                       for y in range(2) for z in range(2)]).reshape(2,2,2)

    for y in range(2):
        for z in range(2):
            assertDensity(P_XT_YZ[:, :, y, z])
            assertDensity(P_X_YZ [:, y, z])
            assertDensity(P_T_YZ [:, y, z])
    print(P_XT_YZ)
    prod_xt_yz = np.array([P_X_YZ[x, y, z] * P_T_YZ[t, y, z] for x, t, y, z in
                     itertools.product(*map(range, [2,2,2,2]))]).reshape(2,2,2,2)
    print(prod_xt_yz)
    print(np.all(np.abs(P_XT_YZ - prod_xt_yz) < .001))

    #indépendance X et (Y,Z)
    P_XYZ = np.array([np.sum(P_XYZT[x,y,z,:]) for x,y,z in
                      itertools.product(*map(range, [2,2,2]))]).reshape(2,2,2)
    P_X = np.array([np.sum(P_XYZ[x,:,:]) for x in range(2)])

    P_YZ2 = np.array([np.sum(P_XYZ[:,y,z]) for y,z in
                      itertools.product(*map(range, [2,2]))]).reshape(2,2)

    prod_xyz = np.array([P_X[x] * P_YZ2[y, z] for x, y, z in
                         itertools.product(*map(range, [2,2,2]))]).reshape(2,2,2)
    print(P_XYZ)
    print(prod_xyz)
    print(np.all(np.abs(P_XYZ - prod_xyz) < .001))

def read_file(filename):
    """
    Renvoie les variables aléatoires et la probabilité contenues dans le
    fichier dont le nom est passé en argument.
    """
    Pjointe = gum.Potential ()
    variables = []

    fic = open ( filename, 'r' )
    # on rajoute les variables dans le potentiel
    nb_vars = int ( fic.readline () )
    for i in range ( nb_vars ):
        name, domsize = fic.readline ().split ()
        variable = gum.LabelizedVariable(name,name,int (domsize))
        variables.append ( variable )
        Pjointe.add(variable)

    # on rajoute les valeurs de proba dans le potentiel
    cpt = []
    for line in fic:
        cpt.append ( float(line) )
    Pjointe.fillWith(np.array ( cpt ) )

    fic.close ()
    return np.array ( variables ), Pjointe

def conditional_indep(p, x, y, z, e):
    """p:potential, x,y:labelized var, z:labelized var list, e:float
    return X indep Y | Z"""
    xyz = p.margSumIn([i.name() for i in [x,y,*z]])
    yz = xyz.margSumIn([i.name() for i in [y,*z]])
    xz = xyz.margSumIn([i.name() for i in [x,*z]])
    z = xz.margSumIn([i.name() for i in z])
    x_z = xz / z
    y_z = yz / z

    q = xyz - x_z*y_z

    return q.abs().max() < e

def compact_conditional_proba(p,x,e):
    """p:potential, x:labelized var, e:float"""
    k = p.variablesSequence()
    for xi in k:
        k.remove(xi)
        if not conditional_indep(p,x,xi,k,e):
            k.append(xi)
    q = p.margSumIn([i.name() for i in k])
    return p.margSumIn([x.name()])/q

def consoMemoire():
    var, proba = read_file("asia.txt")
    gnb.showPotential(proba)
    c = compact_conditional_proba(proba, var[0],0.001)
    gnb.showPotential(c)

def main():
    #galton()
    #visualisationIndependances()
    #independancesConditionnelles()
    consoMemoire()

main()