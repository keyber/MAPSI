import numpy as np
import matplotlib.pyplot as plt
import math

def read_file(filename):
    """
    Lit un fichier USPS et renvoie un tableau de tableaux d'images.
    Chaque image est un tableau de nombres réels.
    Chaque tableau d'images contient des images de la même classe.
    Ainsi, T = read_file ( "fichier" ) est tel que T[0] est le tableau
    des images de la classe 0, T[1] contient celui des images de la classe 1,
    et ainsi de suite.
    """
    # lecture de l'en-tête
    infile = open(filename, "r")
    nb_classes, nb_features = [int(x) for x in infile.readline().split()]
    
    # creation de la structure de données pour sauver les images :
    # c'est un tableau de listes (1 par classe)
    data = np.empty(10, dtype=object)
    filler = np.frompyfunc(lambda x: list(), 1, 1)
    filler(data, data)
    
    # lecture des images du fichier et tri, classe par classe
    for ligne in infile:
        champs = ligne.split()
        if len(champs) == nb_features + 1:
            classe = int(champs.pop(0))
            data[classe].append(list(map(lambda x: float(x), champs)))
    infile.close()
    
    # transformation des list en array
    output = np.empty(10, dtype=object)
    filler2 = np.frompyfunc(lambda x: np.asarray(x), 1, 1)
    filler2(data, output)
    
    return output


def display_image(X):
    """
    Etant donné un tableau X de 256 flotants représentant une image de 16x16
    pixels, la fonction affiche cette image dans une fenêtre.
    """
    # on teste que le tableau contient bien 256 valeurs
    if X.size != 256:
        raise ValueError("Les images doivent être de 16x16 pixels")
    
    # on crée une image pour imshow: chaque pixel est un tableau à 3 valeurs
    # (1 pour chaque canal R,G,B). Ces valeurs sont entre 0 et 1
    Y = X / X.max()
    img = np.zeros((Y.size, 3))
    for i in range(3):
        img[:, i] = X
    
    # on indique que toutes les images sont de 16x16 pixels
    img.shape = (16, 16, 3)
    
    # affichage de l'image
    plt.imshow(img)
    plt.show()

try:
    training_data = read_file("usps_train.txt")
except FileNotFoundError:
    training_data = read_file("2015_tme3_usps_train.txt")

def learnML_class_parameters(classData):
    """pour une classe,
    retourne pour chaque pixel
    la moyenne et la variance de ses niveaux de gris
    sous la forme (m_256, s2_256)
    """
    return np.mean(classData, axis=0), np.var(classData, axis=0)

def learnML_all_parameters(trainingData):
    """return  liste (m_256 s2_256)"""
    return [learnML_class_parameters(i) for i in trainingData]

def log_likelihood(image, classParam):
    """probabilté d'avoir cette image sousvl'hypothèse
    qu'on suive une loi normale de paramètres fournis"""
    m = classParam[0]
    s2 = classParam[1]
    #s=0 => p=1 => log(p)=0
    return -sum([math.log(2 * math.pi * s2[i]) / 2
                 + ((image[i] - m[i]) ** 2) / s2[i] / 2
                 for i in range(len(image)) if s2[i]])

print('input', [i.shape for i in training_data])
param = learnML_all_parameters(training_data)


try:
    test_data = read_file("usps_test.txt")
except FileNotFoundError:
    test_data = read_file("2015_tme3_usps_test.txt")
print(log_likelihood(test_data[2][3], param[1]))

print([log_likelihood(test_data[0][0], param[i]) for i in range(10)])


def log_likelihoods(image, allParam):
    """vraisemblance de l'image pour chaque classe"""
    return np.array([log_likelihood(image, i) for i in allParam])

print(log_likelihoods(test_data[1][5], param))

def classify_image(image, allParam):
    """détermine la classe de l'image (le chiffre)
    par maximum de vraisemblance"""
    return np.argmax(log_likelihoods(image, allParam))


print(classify_image(test_data[1][5], param))
print(classify_image(test_data[4][1], param))


def classify_all_images(test_data, param):
    """retourne les fréquences de chaque couple
    (classe réelle, classe devinée)"""
    res = np.zeros((10,10))
    for i in range(10):
        aff=True
        for image in test_data[i]:
            #trouve la classe
            j = classify_image(image,param)
            
            #compte
            res[i][j]+=1
            
            #i=j si réussite
            if i!=j and aff:
                aff=False
                print("error",i,j)
                display_image(test_data[i][j] )
                
        #normalise
        res[i] /= res[i].sum()
    return res

from mpl_toolkits.mplot3d import Axes3D#"unused" mais nécessaire

def dessine(classified_matrix):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.linspace ( 0, 9, 10 )
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, classified_matrix, rstride = 1, cstride=1)
    plt.show()

dessine(classify_all_images(test_data, param))

def generate(classParam):
    from numpy.random import normal
    image = np.zeros(256)
    for i in range(256):
        image[i] = normal(classParam[0][i], classParam[1][i]/10)
    return image

print("génération d'images")
for p in param:
    display_image(generate(p))