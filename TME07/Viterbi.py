import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
# truc pour un affichage plus convivial des matrices numpy
np.set_printoptions(precision=2, linewidth=320)
plt.close('all')


def initGD(X,N):
    """entrée: ensemble de séquences d'observations
    sortie: ensemble des séquences d'états"""
    return [np.floor(np.linspace(0, N - .00000001, len(x))).astype(int) for x in X]

def learnHMM(allx, allq, N, K, initTo0=False):
    if initTo0:
        A = np.zeros((N,N))
        B = np.zeros((N,K))
        Pi = np.zeros(N)
    else:
        eps = 1e-8
        A = np.ones((N,N))*eps
        B = np.ones((N,K))*eps
        Pi = np.ones(N)*eps
    

def discretise(X, d):
    """regroupe les rotations en d paquets"""
    intervalle = 360/d
    return np.array([np.floor(xi / intervalle).astype(int) for xi in X])

def main():
    with open('lettres.pkl', 'rb') as f:
        data = pkl.load(f, encoding='latin1')
    X = np.array(data.get('letters'))
    Y = np.array(data.get('labels'))
    nCl = 26
    
    K = 10  # discrétisation (=10 observations possibles)
    N = 5  # 5 états possibles (de 0 à 4 en python)
    Xd = discretise(X, K)

    Pi, A, B = learnHMM(Xd[Y == 'a'], q[Y == 'a'], N, K)
main()