"""fichier auxilliaire"""
import numpy
import matplotlib.pyplot
import math

def compacte(vals, n):
    """retourne le tuple :
    - coo x des barres
    - valeurs sommées par paquet de n
    usage: matplotlib.pyplot.bar(*compacte(vals,n), width=n)"""
    return (
        # décale de n/2, les barres sont centrées sinon
        range(round(n/2), len(vals)+round(n/2), n),
        #somme les valeurs, le dernier paquet n'est pas forcement de taille n
        numpy.array([sum(vals[i:min(i+n, len(vals)-1)]) for i in range(0, len(vals), n)]))

def pltbarCompacte(vals, n):
    """effet de bord sur matplotlib.pyplot
    crée les barres et affiche toutes les graduations"""
    matplotlib.pyplot.bar(*compacte(vals, n), width=n)
    matplotlib.pyplot.xticks(range(0, len(vals), n),range(0, len(vals), n))

def assertDensity(a):
    if abs(numpy.sum(a)-1)>.001 or not areProba(a):
        print(a.shape, a.size)
        print(a)
        print(numpy.sum(a))
        print(max(a))
        assert False

def areProba(a):
    return not numpy.any((a<0)|(a>1))

def correlation(valsA, valsB, pab):
    """:param pab loi jointe de a et b
    :param valsA valeurs prises par la variable aléatoire A
    :param valsB valeurs prises par la variable aléatoire B"""
    assert (*valsA.shape, *valsB.shape) == pab.shape
    assert areProba(pab)
    pa = numpy.array([numpy.sum(pab[i,:]) for i in range(len(valsA))])
    assertDensity(pa)
    #espérance: somme des x * P(x)
    moyA = numpy.sum(pa*valsA)
    #variance: somme des P(x) * (x-moyA)**2
    stdA = math.sqrt(sum([pa[i] * (valsA[i] - moyA) **2 for i in range(len(valsA))]))
    pb = numpy.array([numpy.sum(pab[:,i]) for i in range(len(valsB))])
    assertDensity(pb)
    moyB = numpy.sum(pb*valsB)
    stdB = math.sqrt(sum([pb[i] * (valsB[i] - moyB) **2 for i in range(len(valsB))]))
    #print(moyA, stdA, pa)
    #print(moyB, stdB, pb)

    #covariance = somme des p * (x-E(x))(y-E(y))
    covariance = 0
    for i in range(len(valsA)):
        for j in range(len(valsB)):
            covariance += pab[i][j] * (valsA[i] - moyA) * (valsB[j] - moyB)

    #coef corrélation = covariance / ecarttype1*ecarttype2
    correl = covariance / (stdA*stdB)
    assert abs(correl)<=1.001
    return covariance, correl

def correlationBrute(valsA, valsB):
    """ecart type de A et B doivent être non nuls"""
    assert len(valsA)==len(valsB)
    n = len(valsA)
    moyA = numpy.mean(valsA)
    stdA = numpy.std(valsA)
    moyB = numpy.mean(valsB)
    stdB = numpy.std(valsB)
    covariance = 0
    for i in range(n):
        covariance += 1/n * (valsA[i] - moyA) * (valsB[i] - moyB)
    correl = covariance / (stdA*stdB)
    assert abs(correl)<=1.001
    return covariance, correl

def correlationPVar(pA_B, pB, varB):
    """P[i] = [1 avec proba p[i], 0 avec proba 1-p[i]] <-> varB[i] = p[i]. dem:
    p(Alt=alt, VD=vd) = p(VD=d|Alt=a)*p(Alt=a)
     la covariance est la somme sur a de
         P   * p(a) * (a-am) * (1-pm)
     + (1-P) * p(a) * (a-am) * (0-pm)
     = p(a)*(a-am) * [(1-pm)*P - dm*(1-P)]
     = p(a)*(a-am) * (P - pmP - pm + pmP)
     = p(a)*(a-am) * (P - pm)"""
    assert pA_B.ndim==1 and pA_B.shape==pB.shape and pB.shape==varB.shape
    pab = numpy.array([pA_B*pB, [(1 - pA_B[i]) * pB[i] for i in range(len(pA_B))]])
    return correlation(numpy.array([1,0]), varB, pab)
