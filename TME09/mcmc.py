import random
import matplotlib.pyplot as plt
import numpy as np

def estimationPi():
    def tirage(f):
        return random.uniform(-f,f),random.uniform(-f,f)
    
    def monteCarlo(N):
        """nombre tirage
        :return p, x, y"""
        l = np.array([tirage(1) for _ in range(N)])
        return (sum(p[0]*p[0] + p[1]*p[1] < 1 for p in l)/N,
                l[:,0], l[:,1])
    
    plt.figure()
    
    # trace le carré
    plt.plot([-1, -1, 1, 1], [-1, 1, 1, -1], '-')
    
    # trace le cercle
    x = np.linspace(-1, 1, 100)
    y = np.sqrt(1- x*x)
    plt.plot(x, y, 'b')
    plt.plot(x, -y, 'b')
    
    # estimation par Monte Carlo
    pi, x, y = monteCarlo(int(1e4))
    
    # trace les points dans le cercle et hors du cercle
    dist = x*x + y*y
    plt.plot(x[dist <=1], y[dist <=1], "g.")
    plt.plot(x[dist>1], y[dist>1], "r.")
    plt.show()

import pickle as pkl

# si vos fichiers sont dans un repertoire "ressources"
with open("countWar.pkl", 'rb') as f:
    count, mu, A = pkl.load(f, encoding='latin1')

with open("secret.txt", 'r') as f:
    secret = f.read()[:-1]  # -1 pour supprimer le saut de ligne

with open("secret2.txt", 'r') as f:
    secret2 = f.read()[:-1]  # -1 pour supprimer le saut de ligne


def swapF(d):
    """permutte deux éléments distincts tirés aléatoirement"""
    a,b = random.sample(list(d), 2)
    res = dict(d)
    tmp = res[a]
    res[a]=res[b]
    res[b]=tmp
    return res


def decrypt(s, d):
    return "".join((d[c] for c in s))

tau = {'a' : 'b', 'b' : 'c', 'c' : 'a', 'd' : 'd' }
assert decrypt ( "aabcd", tau ) == 'bbcad'
assert decrypt ( "dcba", tau ) == 'dacb'


#chars2index = dict(zip(np.array(list(count.keys())), np.arange(len(count.keys()))))
with open("fichierHash.pkl", 'rb') as f:
    chars2index = pkl.load(f, encoding='latin1')
    
def logLikelihood(s,m,a,d):
    """s:chaine
    :return logvraisemblance de s par rapport au modèle bigramme mu, A"""
    return np.log(m[d[s[0]]]) + sum(np.log(a[d[s[i-1]], d[s[i]]]) for i in range(1,len(s)))

def metropolisHastings(s,m,a,t,N,chars2index):
    """"""
    tcurr = t
    decrypted = decrypt(s,tcurr)
    lcurr = logLikelihood(decrypted,m,a,chars2index)
    bestSol = (decrypted,lcurr,t)
    
    #itère N fois et garde la meilleure solution rencontrée
    for _ in range(N):
        t = swapF(tcurr)
        decrypted = decrypt(s,t)
        l = logLikelihood(decrypted,m,a,chars2index)
        
        if l > bestSol[1]:
            bestSol = decrypted,l,t
        
        borne = np.exp(l-lcurr)
        if borne>=1 or random.random()< borne:
            tcurr = t
            lcurr = l
    return bestSol
    
assert abs(logLikelihood("abcd", mu, A, chars2index) + 24.600258560804818) < 1e-9
assert abs(logLikelihood("dcba", mu, A, chars2index) + 26.274828997400395) < 1e-9

def identityTau(count):
    tau = {}
    for k in list(count.keys ()):
        tau[k] = k
    return tau

#print(metropolisHastings(secret2,mu,A,identityTau(count),10000,chars2index))

# ATTENTION: mu = proba des caractere init, pas la proba stationnaire
# => trouver les caractères fréquents = sort (count) !!
# distribution stationnaire des caracteres
freqKeys = np.array(list(count.keys()))
freqVal  = np.array(list(count.values()))
# indice des caracteres: +freq => - freq dans la references
rankFreq = (-freqVal).argsort()

# analyse mess. secret: indice les + freq => - freq
# ATTENTION: 37 cles dans secret, 77 en général...
cles = np.array(list(set(secret))) # tous les caracteres de secret
rankSecret = np.argsort(-np.array([secret.count(c) for c in cles]))
# On ne code que les caractères les plus frequents de mu, tant pis pour les autres
# alignement des + freq dans mu VS + freq dans secret
tau_init = dict([(cles[rankSecret[i]], freqKeys[rankFreq[i]]) for i in range(len(rankSecret))])
print(metropolisHastings(secret, mu, A, tau_init, 2000, chars2index))

cles = np.array(list(set(secret2)))
rankSecret = np.argsort(-np.array([secret2.count(c) for c in cles]))
tau_init = dict([(cles[rankSecret[i]], freqKeys[rankFreq[i]]) for i in range(len(rankSecret))])
print(metropolisHastings(secret2, mu, A, tau_init, 2000, chars2index))

print("c'est normalement le même message, augmenter le nombre d'itérations"
      "pour plus de précision")
#c'est le même message