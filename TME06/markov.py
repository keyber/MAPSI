import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

# affichage d'une lettre
def tracerLettre(let):
    a = -let*np.pi/180 # conversion en rad
    coord = np.array([[0, 0]]) # point initial
    for i in range(len(a)):
        x = np.array([[1, 0]])
        rot = np.array([[np.cos(a[i]), -np.sin(a[i])],[ np.sin(a[i]),np.cos(a[i])]])
        xr = x.dot(rot) # application de la rotation
        coord = np.vstack((coord,xr+coord[-1,:]))
    plt.figure()
    plt.plot(coord[:,0],coord[:,1])
    plt.savefig("exlettre.png")
    return

def discretise(X, d):
    """regroupe les rotations en d paquets"""
    intervalle = 360/d
    return np.array([np.floor(xi / intervalle).astype(int) for xi in X])

def groupByLabel(y):
    index = []
    for i in np.unique(y): # pour toutes les classes
        ind, = np.where(y==i)
        index.append(ind)
    return index

def learnMarkovModel(Xc, d, fillWithOnes):
    """Xc ensemble de signaux d'une classe
    return proba initiale et matrice de passage
    obtenus par max vraisemblance"""
    if fillWithOnes:
        A = np.ones((d, d))
        Pi = np.ones(d)
    else:
        A = np.zeros((d, d))
        Pi = np.zeros(d)
        
    #comptage
    for X in Xc:
        s1 = X[0]
        Pi[s1]+=1
        for s2 in X[1:]:
            A[s1][s2]+=1
            s1=s2
    
    A = A / np.maximum(A.sum(1).reshape(d, 1), 1)  # normalisation
    Pi = Pi / Pi.sum()
    return Pi, A

def probaSequence(seq,Pi,A):
    """return log P(s|Pi,A)"""
    state1 = seq[0]
    som = np.log(Pi[state1])
    for state2 in seq[1:]:
        som += np.log(A[state1][state2])
        state1=state2
    return som


def separeTrainTest(y, pc):
    """separation app/test,
    pc=ratio de points en apprentissage"""
    indTrain = []
    indTest = []
    for i in np.unique(y):  # pour toutes les classes
        ind, = np.where(y == i)
        n = len(ind)
        indTrain.append(ind[np.random.permutation(n)][:int(np.floor(pc * n))])
        indTest.append(np.setdiff1d(ind, indTrain[-1]))
    return indTrain, indTest

def calcPerf(prediction, expected):
    """prediction : liste des classes prédites
    Y: liste des vraies classes"""
    #liste contenant num là où Y contient lettre correspondant
    Ynum = np.empty(expected.shape)
    
    #pour chaque classe ('a', 'b', ...)
    #(associée à son indice par enumerate)
    for num, char in enumerate(np.unique(expected)):
        #matrice des indices dans Y de cette lettre
        matIndicesChar = expected == char
        #met dans Ynum
        Ynum[matIndicesChar] = num
    
    return np.where(prediction != Ynum, 0., 1.).mean()

def precision(X, Y, d, classes, confusion, zeroOrOne):
    """sépare aléatoirement les données en train test
    initialise la matrice des effectifs à zeroOrOne
    ajoute les associations à la matrice de confusion
    retourne la précision"""
    #regroupe les changements d'angles en d groupes
    #application de la discrétisation
    Xd = discretise(X, d)
    
    #sépare le jeu de données en apprentissage et test pour chaque classe
    itrain, itest_classe = separeTrainTest(Y, 0.8)
    
    itestRegrouped = []
    for itestCi in itest_classe:
        itestRegrouped += itestCi.tolist()
    itestRegrouped = np.array(itestRegrouped)
    
    #print("learn model 0 : ", *learnMarkovModel(Xd[index[0]], d))
    
    #parcours de toutes les classes et optimisation des modèles
    models = [learnMarkovModel(Xd[itrain[c]], d, zeroOrOne) for c in classes]
    
    #print([probaSequence(Xd[0], model[0], model[1]) for model in models])
    
    #matrice contenant pour chaque test, la log-vraisemblance de chaque classe
    proba = np.array([[probaSequence(seq, *models[c])
                       for c in classes] for seq in Xd[itestRegrouped]])
    
    prediction = proba.argmax(1)  #max ligne par ligne
    
    expectedChars = Y[itestRegrouped]
    
    d = {char: num for num, char in enumerate(np.unique(expectedChars))}
    expectedInteger = [d[c] for c in expectedChars]
    for expect,pred in zip(expectedInteger,prediction):
        confusion[expect][pred] += 1
        
    return calcPerf(prediction, expectedChars)
    
        
def main():
    with open('lettres.pkl', 'rb') as f:
        data = pkl.load(f, encoding='latin1')
    
    X = np.array(data.get('letters'))  # récupération des données sur les lettres
    Y = np.array(data.get('labels'))  # récupération des étiquettes associées
    
    #groupement des signaux par classe
    #liste des indices dans X des objets de la classe c pour tout c
    #index = groupByLabel(Y) #recalculé par separeTrainTest
    
    classes = range(len(np.unique(Y)))
    #les classes n'ont pas le même nombre d'éléments
    
    #si on apprend et teste avec les même séquences on obtient
    #91% avec 20 états et 69% avec 3 états
    confusion = np.zeros((26, 26))
    
    for nparam, N in zip([[2,3,4,5,6,7,8,9, 10], [10,20,40,80,160,320]], [4,2]):
        plt.figure()
        plt.xlabel('nb catégories')
        plt.ylabel('précision')
        for zeroOrOne in [0,1]:
            lPerf=[]
            #paramètre de discrétisation
            for d in nparam:
                tmp=[precision(X,Y,d,classes,confusion,zeroOrOne) for _ in range(N)]
                lPerf.append([d, np.mean(tmp), np.std(tmp)*2/np.sqrt(N)])
            lPerf = np.array(lPerf)
            
            plt.errorbar(lPerf[:,0], lPerf[:,1], lPerf[:,2], fmt='C'+str(zeroOrOne))
    
    #transforme le tableau de contingence en proba conditionelle à classe réelle
    for i in range(len(confusion)):
        confusion[i]/=np.sum(confusion[i])
        
    plt.figure()
    plt.imshow(confusion, interpolation='nearest')
    plt.colorbar()
    plt.xticks(np.arange(26), np.unique(Y))
    plt.yticks(np.arange(26), np.unique(Y))
    plt.ylabel(u'Vérité terrain')
    plt.xlabel(u'Prédiction')
    plt.savefig("mat_conf_lettres.png")
    plt.show()
    
main()
