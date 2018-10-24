import random
import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
import pyAgrum as gum
import pyAgrum.lib.ipython as gnb
from mpl_toolkits.mplot3d import Axes3D#"unused" mais nécessaire


def assertDensity(a):
    if abs(np.sum(a) - 1) > .001 or not areProba(a):
        print(a.shape, a.size)
        print(a)
        print(np.sum(a))
        print(max(a))
        assert False


def areProba(a):
    return not np.any((a < 0) | (a > 1))


def bernoulli(p):
    return random.random() < p


def binomiale(n, p):
    return sum([bernoulli(p) for _ in range(n)])


def galton():
    """planche de Galton"""
    nbEtage = 10
    arr = np.array([binomiale(nbEtage, .5) for _ in range(100)])
    nbVals = nbEtage + 1
    plt.hist(arr, bins=range(nbVals))
    plt.title('galton')
    plt.show()
    plt.close()


def normale(k, std):
    """variable continue, retourne quelques évalutations de la fonction de densité,
    ne somme pas a 1"""
    if k % 2 == 0:
        raise ValueError('le nombre k doit etre impair')
    
    N_0_STD = lambda x: math.exp(-x ** 2 / 2) / math.sqrt(2 * math.pi)
    x = np.linspace(-2 * std, 2 * std, k)
    y = np.array([N_0_STD(i) for i in x])
    assert areProba(y)
    return x + 2 * std, y


def proba_affine(k, slope):
    """variable continue, retourne quelques évalutations de la fonction de densité,
    ne somme pas a 1"""
    if k % 2 == 0:
        raise ValueError('le nombre k doit etre impair')
    cst = 1 / k - (k - 1) * slope / 2
    
    if cst < 0 or cst + (k - 1) * slope > 1:
        raise ValueError('mauvaise pente' + slope)
    
    x = np.linspace(0, k - 1, k)
    y = np.array([cst + i * slope for i in x])
    assert areProba(y)
    return x, y


def pXY(x, y):
    """retourne le tableau p(x,y) des probas jointes, suppose indépendance
    x horizontal, y vertical"""
    res = np.array([y * valx for valx in x])
    assert areProba(res)
    return res


def dessine(P_jointe):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x = np.linspace(0, 10, P_jointe.shape[1])
    y = np.linspace(0, 10, P_jointe.shape[0])
    X, Y = np.meshgrid(x, y)
    # print(X.shape, Y.shape, P_jointe.shape)
    ax.plot_surface(X, Y, P_jointe)
    ax.set_xlabel('A')
    ax.set_ylabel('B')
    ax.set_zlabel('P(A) * P(B)')
    plt.show()
    plt.close()


def visualisationIndependances():
    """Visualisation d'indépendances"""
    normx, normy = normale(55, 2)
    plt.plot(normx, normy)
    
    affx, affy = proba_affine(7, .04)
    plt.plot(affx, affy)
    
    norm_aff = pXY(normy, affy)
    plt.close()
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
    
    P_YZ = np.array([[np.sum(P_XYZT[:, y, z, :])
                      for z in range(2)] for y in range(2)])
    assertDensity(P_YZ)
    
    #calcul de différentes manières
    # x à l'intérieur
    P_XT_YZ = np.array([P_XYZT[x, y, z, t] / P_YZ[y, z] for x, t, y, z in
                        itertools.product(*map(range, [2, 2, 2, 2]))])\
        .reshape(2, 2, 2, 2)
    
    # x extérieur
    P_X_YZ = np.array([[[np.sum(P_XT_YZ[x, :, y, z]) for z in range(2)]
                        for y in range(2)] for x in range(2)])
    
    # intérieur
    P_T_YZ = np.array([np.sum(P_XT_YZ[:, t, y, z]) for t in range(2)
                       for y in range(2) for z in range(2)]).reshape(2, 2, 2)
    
    for y in range(2):
        for z in range(2):
            assertDensity(P_XT_YZ[:, :, y, z])
            assertDensity(P_X_YZ[:, y, z])
            assertDensity(P_T_YZ[:, y, z])
    prod_xt_yz = np.array([P_X_YZ[x, y, z] * P_T_YZ[t, y, z] for x, t, y, z in
                           itertools.product(*map(range, [2, 2, 2, 2]))])\
        .reshape(2, 2, 2, 2)
    print('XT indép sachant YZ : ', np.all(np.abs(P_XT_YZ - prod_xt_yz) < .001))
    
    
    # indépendance X et (Y,Z)
    P_XYZ = np.array([np.sum(P_XYZT[x, y, z, :]) for x, y, z in
                      itertools.product(*map(range, [2, 2, 2]))]).reshape(2, 2, 2)
    P_X = np.array([np.sum(P_XYZ[x, :, :]) for x in range(2)])
    
    P_YZ2 = np.array([np.sum(P_XYZ[:, y, z]) for y, z in
                      itertools.product(*map(range, [2, 2]))]).reshape(2, 2)
    
    prod_xyz = np.array([P_X[x] * P_YZ2[y, z] for x, y, z in
                         itertools.product(*map(range, [2, 2, 2]))]).reshape(2, 2, 2)
    print('X indép YZ : ', np.all(np.abs(P_XYZ - prod_xyz) < .001))


def read_file(filename):
    """Renvoie les variables aléatoires et la probabilité contenues dans le
    fichier dont le nom est passé en argument."""
    Pjointe = gum.Potential()
    variables = []
    
    fic = open(filename, 'r')
    # on rajoute les variables dans le potentiel
    nb_vars = int(fic.readline())
    for i in range(nb_vars):
        name, domsize = fic.readline().split()
        variable = gum.LabelizedVariable(name, name, int(domsize))
        variables.append(variable)
        Pjointe.add(variable)
    
    # on rajoute les valeurs de proba dans le potentiel
    cpt = []
    for line in fic:
        cpt.append(float(line))
    Pjointe.fillWith(np.array(cpt))
    
    fic.close()
    return np.array(variables), Pjointe


def conditional_indep(p, x, y, z, e):
    """p potential, x,y:labelized var, z:variable sequence, e:float
    return  true if X indep Y | Z
    suppose x et y pas dans z"""
    #calcule les marginales puis compare leur produit avec la loi jointe
    
    #l'ensemble du "sachant" est vide
    if len(z) == 0:
        xy = p.margSumIn([x.name(), y.name()])
        x = xy.margSumIn([x.name()])
        y = xy.margSumIn([y.name()])
        q = xy - x * y
    else:
        #margSumIn ne garde que les dimensions spécifiées (en sommant les autres)
        xyz = p.margSumIn([i.name() for i in [x, y, *z]])
        #margSumOut les enlève (en sommant les autres)
        xz = xyz.margSumOut([y.name()])
        yz = xyz.margSumOut([x.name()])
        z = xz.margSumOut([x.name()])
        xy_z = xyz / z
        x_z = xz / z
        y_z = yz / z
        q = xy_z - x_z * y_z
        
    return q.abs().max() < e
    

def compact_conditional_proba(p, X, e):
    """p:potential = loi jointe des Xi,
    X:labelized var, e:float
    return P(X|{Xk}) avec Xk minimum sans perdre d'information
    """
    #liste_sachant initialisé à liste des variables de p (sans X)
    liste_sachant = p.variablesSequence()  # type: list
    liste_sachant.remove(X)
    
    print('pour', X.name(), 'à', e, 'près, il est équivalent de savoir :')
    print('-', *map(lambda x: x.name(), liste_sachant), 'ou')
    
    #enlève les Xi un par un si X indep Xi | liste_sachant\Xi
    for xi in liste_sachant[:]:#[:] nécessaire car la liste est modifiée
        #enlève Xi pour déterminer si X indep Xi, puis le remet si nécessaire
        ind = liste_sachant.index(xi)
        liste_sachant.remove(xi)
        if not conditional_indep(p, X, xi, liste_sachant, e):
            liste_sachant.insert(ind, xi)
    
    print('-',*map(lambda x: x.name(), liste_sachant))
    print('gain :', len(p.variablesSequence()) - 1, '->', len(liste_sachant), '\n')
    
    #loi jointe
    r = p.margSumIn([i.name() for i in liste_sachant + [X]])
    
    #divise par liste sachant pour obtenir conditionelle
    if len(liste_sachant) != 0:
        r = r / p.margSumIn([i.name() for i in liste_sachant])
    
    return r
    

def create_bayesian_network(p, e):
    """proba jointe à compacter avec une tolérance de e
    return la loi jointe P(X0...Xi) sous la forme d'un
    produit de probas conditionnelles :
    [ P(X0) , P(X1|X0) , ... , P(Xn|X0...Xn-1) ]
    """
    liste = []
    for _ in range(len(p.variablesSequence())-1):
        #P(X0...Xi) = P(Xi|X0...Xi-1) * P(X0...Xi-1)
        #calcule et ajoute P(Xi|X0...Xi-1)
        liste.append(compact_conditional_proba(p, p.variablesSequence()[-1], e))
        
        #on calcule P(X0...Xi-1) par marginalisation
        #et on réitère le processus
        p = p.margSumOut([p.variablesSequence()[-1].name()])
        
    #ajoute P(X0)
    liste.append(p)
    
    #met P(X0) au début et P(Xn|X0...Xn-1) à la fin
    liste.reverse()
    return liste

def proba_from_bn(bn, x):
    """x de la forme
    {nom: val for nom, val in zip(p.var_names, [1,0,1,1...])}"""
    return np.prod([ssproba[x] for ssproba in bn])
    

def consoMemoire():
    """
    bn = gum.loadBN("munin1.bif")#marche mais n'est pas du meme type que read_file
    #affichage de la taille des probabilités jointes compacte et non compacte
    print(bn)
    #affichage graphique du réseau bayésien, ne fonctionne pas
    #import pyAgrum.lib.notebook as gnb
    #gnb.showBN(bn)#pydotplus.graphviz.InvocationException: GraphViz's executables not found
    proba = bn.completeInstantiation()
    """
    try:
        _, proba = read_file("asia.txt")
        gnb.showPotential(proba.margSumIn(proba.var_names[:2]))
    except FileNotFoundError:
        _, proba = read_file("2017_tme2_asia.txt")
    print('\n\n')
    
    for e in [.1, .01, .0001]:
        bn = create_bayesian_network(proba, e)
        
        print('à', e, 'près ON PASSE DE', proba.toarray().size,
              'paramètres à', sum([i.toarray().size for i in bn]))
        
        #reconstitue la loi jointe et compare avec la loi initiale pour tester
        p2 = bn[0]#shallow copies, invalide bn
        for i in bn[1:]:
            p2 *= i
        print("ERREUR MAXIMALE en reconstruisant la loi jointe:", (p2-proba).abs().max(), '\n\n')
        
        
def main():
    galton()
    visualisationIndependances()
    independancesConditionnelles()
    consoMemoire()

main()
