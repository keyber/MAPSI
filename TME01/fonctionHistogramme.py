"""fichier auxilliaire"""
import numpy as np
import matplotlib.pyplot as plt

def compacte(vals, n):
    """retourne le tuple :
    - coo x des barres
    - valeurs sommées par paquet de n
    usage: plt.bar(*compacte(vals,n), width=n)"""
    return (
        # décale de n/2, les barres sont centrées sinon
        range(round(n/2), len(vals)+round(n/2), n),
        #somme les valeurs, le dernier paquet n'est pas forcement de taille n
        np.array([sum(vals[i:min(i+n, len(vals)-1)]) for i in range(0, len(vals), n)]))

def pltbarCompacte(vals, n):
    """effet de bord sur plt
    crée les barres et affiche toutes les graduations"""
    plt.bar(*compacte(vals, n), width=n)
    plt.xticks(range(0, len(vals), n),range(0, len(vals), n))
