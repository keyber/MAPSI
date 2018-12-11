import random
import numpy as np
import matplotlib.pyplot as plt

def genLin(N, a,b,sig):
    X = []
    Y = []
    for _ in range(N):
        x=random.random()
        X.append(x)
        Y.append(a*x + b + random.gauss(0,sig))
    return np.array(X), np.array(Y)

def genQuad(N, a,b,c, sig):
    X = []
    Y = []
    for _ in range(N):
        x=random.random()
        X.append(x)
        Y.append(a*x*x + b*x +c + random.gauss(0,sig))
    return np.array(X), np.array(Y)

def resolutionBayesienAB(X, Y):
    """méthode analytique à partir de paramètres probabilistes"""
    mx, my = np.mean(X), np.mean(Y)
    cov = np.cov(X,Y)#matrice variance-covariance
    vx = cov[0][0]#variance de x
    cov=cov[0][1] #covariance(x,y)
    b = cov/vx
    return b, my - b*mx

def resolutionMoindresCarres(X, Y):
    """méthode analytique à partir d'une fonction de coût
    "Least squares" """
    return np.linalg.solve(X.T.dot(X), X.T.dot(Y))

def testResAnalytiques(X,Y):
    n=len(X)
    
    a,b = resolutionBayesienAB(X, Y)
    
    XAB = np.hstack((X.reshape(n, 1), np.ones((n, 1))))
    a2,b2 = resolutionMoindresCarres(XAB, Y)
    
    print(a,b)
    assert abs(a-a2)<1e-5 and abs(b-b2)<1e-5
    
    f0 = b
    f1 = a+b
    plt.plot([0,1],[f0,f1], c='r')
    
    XA = X.reshape(n, 1)
    a3 = resolutionMoindresCarres(XA,Y)
    plt.plot([0,1],[0,a3], c='k')

def resolutionDescenteGradient(X:np.array, Y, eps, nIteration):
    D = np.zeros(X.shape[1])  # init à 0
    listD = [D]
    for _ in range(nIteration):
        grad = 2 * X.T.dot(X.dot(D) - Y)
        D = D.copy() - grad*eps
        listD.append(D)
    
    return np.array(listD)

def traceEspace(X,Y,wstar,intX, intY):
    # tracé de l'espace des couts
    ngrid = 50
    w1range = np.linspace(*intX, ngrid)
    w2range = np.linspace(*intY, ngrid)
    w1, w2 = np.meshgrid(w1range, w2range)
    
    cost = np.array([[np.log(((X.dot(np.array([w1i, w2j])) - Y) ** 2).sum())
                      for w1i in w1range] for w2j in w2range])
    
    plt.contour(w1, w2, cost)
    plt.scatter(wstar[0], wstar[1], c='r')

def trace3D(X,Y,wstar, listW):
    from mpl_toolkits.mplot3d import Axes3D
    ngrid = 20
    w1range = np.linspace(-0.5, 8, ngrid)
    w2range = np.linspace(-1.5, 1.5, ngrid)
    w1, w2 = np.meshgrid(w1range, w2range)

    cost = np.array([[np.log(((X.dot(np.array([w1i, w2j])) - Y) ** 2).sum())
                      for w1i in w1range] for w2j in w2range])


    costPath = np.array([np.log(((X.dot(wtmp) - Y) ** 2).sum()) for wtmp in listW])
    costOpt = np.log(((X.dot(wstar) - Y) ** 2).sum())

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(w1, w2, cost, rstride=1, cstride=1)
    ax.scatter(wstar[0], wstar[1], costOpt, c='r')
    ax.plot(listW[:, 0], listW[:, 1], costPath, 'k+-', lw=3)

def exLin():
    n=100
    X,Y = genLin(n, 6, -1,.4)
    plt.scatter(X, Y)
    testResAnalytiques(X,Y)
    plt.show()
    
    XAB = np.hstack((X.reshape(n, 1), np.ones((n, 1))))
    print("6e-3 converge, 8e-3 diverge")
    astar,bstar = resolutionMoindresCarres(XAB,Y)
    xmin,xmax,ymin,ymax=(astar-1,astar+1,bstar-1,bstar+1)
    for eps, nIt in zip([6e-3,8e-3], [30,20]):
        res = resolutionDescenteGradient(XAB, Y, eps, nIt)
        plt.plot(res[:, 0], res[:, 1], '+-', lw=2)
        xmin,xmax = min(min(res[:,0]),xmin), max(max(res[:,0]),xmax)
        ymin,ymax = min(min(res[:,1]),ymin), max(max(res[:,1]),ymax)
    traceEspace(XAB, Y, (astar,bstar), (xmin,xmax), (ymin,ymax))
    plt.show()
    
    trace3D(XAB, Y, resolutionMoindresCarres(XAB,Y), resolutionDescenteGradient(XAB, Y, 6e-3, 30))
    plt.show()

def addCol(X, listCoefs):
    """retourne la matrice X concaténée à la colonne obtenue comme ne modifie pas X"""
    for coefs in listCoefs:
        col = np.prod(X[:,coefs], axis=1)
        col = np.array(col).reshape((len(X), 1))
        X = np.hstack((X, col))
    return X

def createMatrix(X,ones,squares,cross):
    """ones ajoute un parametre constant au modèle
    squares un paramètre pour chaque carré de variable
    cross un pour chaque produit croisé de variable"""
    XOld=X
    X = XOld.copy()
    if ones:
        X=addCol(X,[[]])
    if squares:
        X=addCol(X,[[i,i] for i in range(XOld.shape[1])])
    if cross:
        X=addCol(X,[[i,j] for i in range(XOld.shape[1]) for j in range(i)])
    return X

def exQuad():
    n=100
    X,Y = genQuad(n, 6, -1, 1,.4)
    plt.scatter(X,Y)
    
    XABC = createMatrix(X.reshape((n,1)),True,True,False)
    #XABC = np.hstack((np.square(X).reshape(n,1), X.reshape(n, 1), np.ones((n, 1))))
    b,c,a = resolutionMoindresCarres(XABC,Y)
    print(a,b,c)
    abscisse = np.linspace(0,1,100)
    f = list(map(lambda x:a*x*x+b*x+c, abscisse))
    plt.plot(abscisse, f, 'r')
    plt.show()

def exRealData():
    data = np.loadtxt("winequality-red.csv", delimiter=";", skiprows=1)
    N, d = data.shape  # extraction des dimensions
    pcTrain = 0.8  # % des données en apprentissage
    nbIt=10
    for choices in [[0,0,0],[1,0,0],[1,1,0],[1,1,1]]:
        moy = 0
        for _ in range(nbIt):
            allindex = np.random.permutation(N)
            indTrain = allindex[:int(pcTrain * N)]
            indTest = allindex[int(pcTrain * N):]
            XTrain = data[indTrain, :-1]  # sans la dernière colonne
            YTrain = data[indTrain, -1]  # dernière colonne (= note à prédire)
            # Echantillon de test (pour la validation des résultats)
            XTest = data[indTest, :-1]  # sans la dernière colonne
            YTest = data[indTest, -1]  # dernière colonne (= note à prédire)
            
            expexted = YTest
            
            XTrainExpanded = createMatrix(XTrain,*choices)
            res = resolutionMoindresCarres(XTrainExpanded,YTrain)
            XTestExpanded  = createMatrix(XTest ,*choices)
            predicted = XTestExpanded.dot(res)
            moy += np.mean(abs(predicted-expexted))/np.mean(expexted)
        print(XTestExpanded.shape[1],"params, moyenne erreur =",moy/nbIt*100,'%')


def main():
    #exLin()
    exQuad()
    exRealData()
    
main()