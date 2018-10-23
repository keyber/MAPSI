import numpy as np


# fonction pour transformer les données brutes en nombres de 0 à n-1
def translate_data(data):
    # création des structures de données à retourner
    nb_variables = data.shape[0]
    nb_observations = data.shape[1] - 1  # - nom variable
    res_data = np.zeros((nb_variables, nb_observations), int)
    res_dico = np.empty(nb_variables, dtype=object)
    
    # pour chaque variable, faire la traduction
    for i in range(nb_variables):
        res_dico[i] = {}
        index = 0
        for j in range(1, nb_observations + 1):
            # si l'observation n'existe pas dans le dictionnaire, la rajouter
            if data[i, j] not in res_dico[i]:
                res_dico[i].update({data[i, j]: index})
                index += 1
            # rajouter la traduction dans le tableau de données à retourner
            res_data[i, j - 1] = res_dico[i][data[i, j]]
    return res_data, res_dico


# fonction pour lire les données de la base d'apprentissage
def read_csv(filename):
    data = np.loadtxt(filename, delimiter=',', dtype=np.str).T
    names = data[:, 0].copy()
    data, dico = translate_data(data)
    return names, data, dico


# etant donné une BD data et son dictionnaire, cette fonction crée le
# tableau de contingence de (x,y) | z
def create_contingency_table(data, dico, x, y, z):
    # détermination de la taille de z
    size_z = 1
    offset_z = np.zeros(len(z))
    j = 0
    for i in z:
        offset_z[j] = size_z
        size_z *= len(dico[i])
        j += 1
    
    # création du tableau de contingence
    res = np.zeros(size_z, dtype=object)
    
    # remplissage du tableau de contingence
    if size_z != 1:
        z_values = np.apply_along_axis(lambda val_z: val_z.dot(offset_z),
                                       1, data[z, :].T)
        i = 0
        while i < size_z:
            indices, = np.where(z_values == i)
            a, b, c = np.histogram2d(data[x, indices], data[y, indices],
                                     bins=[len(dico[x]), len(dico[y])])
            res[i] = (indices.size, a)
            i += 1
    else:
        a, b, c = np.histogram2d(data[x, :], data[y, :],
                                 bins=[len(dico[x]), len(dico[y])])
        res[0] = (data.shape[1], a)
    return res


def sufficient_statistics(data, dico, x, y, z):
    cont_table = create_contingency_table(data,dico,x,y,z)
    
    tailleZ,tailleX,tailleY = len(cont_table),len(dico[x]),len(dico[y])
    
    
    Nxz = [[cont_table[iz][1][ix,:].sum() for ix in range(tailleX)]
           for iz in range(tailleZ)]
    Nyz = [[cont_table[iz][1][:,iy].sum() for iy in range(tailleY)]
           for iz in range(tailleZ)]
    
    res = 0
    for iz in range(tailleZ):
        for ix in range(tailleX):
            for iy in range(tailleY):
                if Nxz[iz][ix]!=0 and Nyz[iz][iy]!=0:
                    a = Nxz[iz][ix] * Nyz[iz][iy] / cont_table[iz][0]
                    res += (cont_table[iz][1][ix,iy] - a) **2 / a
    
    n=0
    for iz in range(tailleZ):
        if cont_table[iz][0]:
            n+=1
    
    dof = (tailleX-1)*(tailleY-1)*n
    return res, dof


def main():
    # names : tableau contenant les noms des variables aléatoires
    # data  : tableau 2D contenant les instanciations des variables aléatoires
    # dico  : tableau de dictionnaires contenant la correspondance (valeur de variable -> nombre)
    names, data, dico = read_csv("asia.csv")
    print(sufficient_statistics(data, dico, 5,2, [1,3,6]))


main()
