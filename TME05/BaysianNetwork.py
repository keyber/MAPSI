import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pydotplus as pydot
import pyAgrum as gum
import pyAgrum.lib.ipython as gnb

#fonction pour transformer les données brutes en nombres de 0 à n-1
from scipy.sparse.data import _data_matrix


def translate_data(data):
    #création des structures de données à retourner
    nb_variables = data.shape[0]
    nb_observations = data.shape[1] - 1  #- nom variable
    res_data = np.zeros((nb_variables, nb_observations), int)
    res_dico = np.empty(nb_variables, dtype=object)
    
    #pour chaque variable, faire la traduction
    for i in range(nb_variables):
        res_dico[i] = {}
        index = 0
        for j in range(1, nb_observations + 1):
            #si l'observation n'existe pas dans le dictionnaire, la rajouter
            if data[i, j] not in res_dico[i]:
                res_dico[i].update({data[i, j]: index})
                index += 1
            #rajouter la traduction dans le tableau de données à retourner
            res_data[i, j - 1] = res_dico[i][data[i, j]]
    return res_data, res_dico


#fonction pour lire les données de la base d'apprentissage
def read_csv(filename):
    data = np.loadtxt(filename, delimiter=',', dtype=np.str).T
    names = data[:, 0].copy()
    data, dico = translate_data(data)
    return names, data, dico


#etant donné une BD data et son dictionnaire, cette fonction crée le
#tableau de contingence de (x,y) | z
def create_contingency_table(data, dico, x, y, z):
    """return l'équivalent de la loi jointe sous la forme :
    pour chaque z : couple
                    (nombre occurence de z,
                     matrice X*Y du nombre d'occcurence de (x,y,z)"""
    
    #détermination de la taille de z
    size_z = 1
    offset_z = np.zeros(len(z))
    j = 0
    for i in z:
        offset_z[j] = size_z
        size_z *= len(dico[i])
        j += 1
    
    #création du tableau de contingence
    res = np.zeros(size_z, dtype=object)
    
    #remplissage du tableau de contingence
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
    """return chi2, Degrees Of Freedom"""
    cont_table = create_contingency_table(data, dico, x, y, z)
    
    tailleZ, tailleX, tailleY = len(cont_table), len(dico[x]), len(dico[y])
    
    #assert tailleZ == np.prod([len(dico[i]) for i in z])
    
    Nxz = [[cont_table[iz][1][ix, :].sum() for ix in range(tailleX)]
           for iz in range(tailleZ)]
    Nyz = [[cont_table[iz][1][:, iy].sum() for iy in range(tailleY)]
           for iz in range(tailleZ)]
    
    chi2 = 0
    for iz in range(tailleZ):
        for ix in range(tailleX):
            for iy in range(tailleY):
                if Nxz[iz][ix] != 0 and Nyz[iz][iy] != 0:
                    #les maths voudraient >= 5
                    a = Nxz[iz][ix] * Nyz[iz][iy] / cont_table[iz][0]
                    chi2 += (cont_table[iz][1][ix, iy] - a) ** 2 / a
    
    n = 0
    for iz in range(tailleZ):
        if cont_table[iz][0]:  #z apparait au moins une fois
            #les maths voudraient >= 5
            n += 1
    
    dof = (tailleX - 1) * (tailleY - 1) * n
    return chi2, dof


def indep_score(data, dico, x, y, z):
    """return (pvalue, dof(qui n'a pas a être utilisé))
    alpha=5% : "si ce qu'on observe
                ~avait moins de 5% de chance d'arriver X~
                fait partie des 5% des cas les plus extrêmes
                on rejette H0".
    pvalue=0.123% : "la probabilité sous H0 d'obtenir un résultat
                     au moins aussi extrême que ce qu'on observe"
    Si alpha < pvalue, on accepte H0 sinon on rejete
    [0, pvalue] correspond à l'ensemble des risques de première espèce
    qu'on aurait pu prendre pour encore considérer H0
    
    H0 : indépendance de x et y sachant z
    plus pvalue est grand, plus on avait de
        chance de faire cette observation sous H0,
        plus x et y on l'air indépendant
    """
    assert x not in z and y not in z
    #normalement on voudrait que le tableau de contingence soit
    #rempli par des valeurs supérieures à 5 (pour chaque couple X,Y,Z)
    #on remplace ici par moyenne supérieure à 5
    
    #énoncé pas rigoureux ne pas faire attention
    dmin = 5 * len(dico[x]) * len(dico[y]) * \
           np.prod([len(dico[i]) for i in z])
    #si on a pas assez de valeur, on suppose la dépendance
    if len(data[0]) < dmin:
        return -1, 1
    chi2, dof = sufficient_statistics(data, dico, x, y, z)
    return stats.chi2.sf(chi2, dof), dof


def best_candidate(data, dico, x, z, a):
    """return indice de y dans le csv
    y: variable à gauche de x la plus dépendante de x sachant z
    (y = argmin(pvalue))
    sous forme de
        une liste d'un entier
        ou liste vide si pvalue > a : X indep tous sachant Z"""
    assert x not in z
    #sachant y, y est indépendant de tout, rien besoin de calculer
    scores = [indep_score(data, dico, i, x, z)[0]
              if i not in z else 1 for i in range(x)]
    ind = np.argmin(scores)
    val = scores[ind]
    if val > a:
        return []
    return [ind]


def create_parents(data, dico, x, a):
    """return liste z des parents de x dans le réseau bayésien
    un parent se trouve forcément à gauche de x"""
    #affichage pour voir si l'exécution avance
    print(x, len(dico))
    
    z = []
    #ajoute les parents un par un
    while True:
        #par ordre de score d'indépendance croissant
        b = best_candidate(data, dico, x, z, a)
        z += b
        if len(b) == 0 or b[0]==0:
            break
    #toutes les autres variables sont indépendantes de x sachant z
    return z


def learn_BN_structure(data, dico, a):
    """return tableau contenant
    pour chaque noeuds, la liste de ses parents"""
    #le premier noeud est forcément sans racine
    return np.array([[]] + [create_parents(data, dico, i, a)
                            for i in range(1, len(dico))])


style = {"bgcolor": "#6b85d1", "fgcolor": "#FFFFFF"}


def display_BN(node_names, bn_struct, bn_name):
    graph = pydot.Dot(bn_name, graph_type='digraph')
    
    #création des noeuds du réseau
    for name in node_names:
        new_node = pydot.Node(name,
                              style="filled",
                              fillcolor=style["bgcolor"],
                              fontcolor=style["fgcolor"])
        graph.add_node(new_node)
    
    #création des arcs
    for node in range(len(node_names)):
        parents = bn_struct[node]
        for par in parents:
            new_edge = pydot.Edge(node_names[par], node_names[node])
            graph.add_edge(new_edge)
    
    #sauvegarde et affaichage
    outfile = bn_name + '.png'
    graph.write_png(outfile)
    img = mpimg.imread(outfile)
    plt.imshow(img)
    plt.show()


def learn_parameters(bn_struct, ficname):
    #création du dag correspondant au bn_struct
    graphe = gum.DAG()
    nodes = [graphe.addNode() for _ in range(bn_struct.shape[0])]
    for i in range(bn_struct.shape[0]):
        for parent in bn_struct[i]:
            graphe.addArc(nodes[parent], nodes[i])
    
    #appel au BNLearner pour apprendre les paramètres
    learner = gum.BNLearner(ficname)
    learner.useScoreLog2Likelihood()
    learner.useAprioriSmoothing()
    return learner.learnParameters(graphe)

DEBUG=False
def main():
    nomFichier = '2014_tme5_asia3.csv'#alarm lag
    #names : tableau contenant les noms des variables aléatoires
    #data  : tableau 2D contenant les
    #instanciations des variables aléatoires
    #dico  : tableau de dictionnaires contenant la
    #correspondance (valeur de variable -> nombre)
    names, data, dico = read_csv(nomFichier)
    
    #data : matrice nbvar * nbMesure
    print(names)
    print(data.shape)
    
    if DEBUG:
        tableau_contingence=np.zeros(2**len(data))
        for i in range(len(data[0])):
            ind=0
            for j in range(len(data)):#[0,1,2,3,6,4,7,5]
                ind*=2
                ind+=data[j][i]
            tableau_contingence[ind]+=1
        tableau_contingence/=len(data[0])

        #print(tableau_contingence)
        jointNaive = gum.Potential()
        for name in names:
            v = gum.LabelizedVariable(name, name, 3)
            print(v)
            jointNaive.add(v)
            print(jointNaive.var_names)
        jointNaive.fillWith(tableau_contingence)
        
        print(jointNaive)
        
    """
    print(sufficient_statistics(data, dico, 5,2, [1,3,6]))
    print( indep_score  ( data, dico, 1, 2,[3, 4]))
    print(best_candidate ( data, dico, 1, [], 0.05 ))
    print(best_candidate ( data, dico, 5, [6], 0.05 ))
    print(create_parents ( data, dico, 1, 0.05 ))
    print(create_parents ( data, dico, 4, 0.05 ))
    print(create_parents ( data, dico, 5, 0.05 ))
    print(create_parents ( data, dico, 6, 0.05 ))"""
    
    bn = learn_BN_structure(data, dico, .05)
    display_BN(names, bn, "BN")
    
    #création du réseau bayésien à la aGrUM
    gumbn = learn_parameters(bn, nomFichier)
    
    #affichage de sa taille
    print(gumbn)
    
    if DEBUG:
        #liste des Conditional Probability Table
        l = [gumbn.cpt(gumbn.idFromName(name)) for name in names]
        #loi jointe
        joint = l[0]
        for cpt in l[1:]:
            joint *= cpt
    
        rawProba = np.array([proba for proba in joint]).reshape(2**len(data))
    

        c=1
        l=[]
        for ind in range(len(data)):
            m=0
            for cpt in range(2**len(data)):
                param={names[i]:(cpt>>i)&1 for i in range(len(data))}
                m=max(m, abs(joint[param] - tableau_contingence[cpt]))
            l.append(m)
            if c%1000==0:
                print(c)
            c+=1
        ind = np.argmin(l)
        print(l[ind], ind)
        
        print("ERREUR MAXIMALE en reconstruisant la loi jointe:",
              np.abs(rawProba - tableau_contingence).max())
    
        print("ERREUR MAXIMALE en reconstruisant la loi jointe:",
              (joint - jointNaive).abs().max())
    
        print("sommes", rawProba.sum() , tableau_contingence.sum())
        print("max", tableau_contingence.max(), rawProba.max())
    
    #gnb.showPotential(gum.getPosterior( gumbn,
    #{'smoking?': 'true', 'tuberculosis?': 'false'},'bronchitis?'))
    
    #theo = [sum(data[i])/len(data[i]) for i in range(len(names))]
    #for i in range(len(names)):
    #    print(gum.getPosterior(gumbn,{},names[i])[{names[i]:1}])
    #    print(sum(data[i])/len(data[i]),'\n')
    #print("errmax des marginales : ",
    #      max([min(abs(r-t), abs(r+t-1)) for t,r in zip(theo, recons)]))
    
main()
