import numpy as np
import matplotlib.pyplot as plt

def read_file ( filename ):
    """
    Lit un fichier USPS et renvoie un tableau de tableaux d'images.
    Chaque image est un tableau de nombres réels.
    Chaque tableau d'images contient des images de la même classe.
    Ainsi, T = read_file ( "fichier" ) est tel que T[0] est le tableau
    des images de la classe 0, T[1] contient celui des images de la classe 1,
    et ainsi de suite.
    """
    # lecture de l'en-tête
    infile = open ( filename, "r" )
    nb_classes, nb_features = [ int( x ) for x in infile.readline().split() ]

    # creation de la structure de données pour sauver les images :
    # c'est un tableau de listes (1 par classe)
    data = np.empty ( 10, dtype=object )
    filler = np.frompyfunc(lambda x: list(), 1, 1)
    filler( data, data )

    # lecture des images du fichier et tri, classe par classe
    for ligne in infile:
        champs = ligne.split ()
        if len ( champs ) == nb_features + 1:
            classe = int ( champs.pop ( 0 ) )
            data[classe].append ( list ( map ( lambda x: float(x), champs ) ) )
    infile.close ()

    # transformation des list en array
    output  = np.empty ( 10, dtype=object )
    filler2 = np.frompyfunc(lambda x: np.asarray (x), 1, 1)
    filler2 ( data, output )

    return output

def display_image ( X ):
    """
    Etant donné un tableau X de 256 flotants représentant une image de 16x16
    pixels, la fonction affiche cette image dans une fenêtre.
    """
    # on teste que le tableau contient bien 256 valeurs
    if X.size != 256:
        raise ValueError ( "Les images doivent être de 16x16 pixels" )

    # on crée une image pour imshow: chaque pixel est un tableau à 3 valeurs
    # (1 pour chaque canal R,G,B). Ces valeurs sont entre 0 et 1
    Y = X / X.max ()
    img = np.zeros ( ( Y.size, 3 ) )
    for i in range ( 3 ):
        img[:,i] = X

    # on indique que toutes les images sont de 16x16 pixels
    img.shape = (16,16,3)

    # affichage de l'image
    plt.imshow( img )
    plt.show ()

training_data = read_file ( "usps_train.txt" )

# affichage du 1er chiffre "2" de la base:
#display_image(training_data[2][0] )

def learnML_class_parameters(classData):
    """return m_256 s2_256"""
    return np.mean(classData, axis=0), np.var(classData, axis=0)

def learnML_all_parameters(trainingData):
    """return  liste (m_256 s2_256)"""
    return [learnML_class_parameters(i) for i in trainingData]

import math
def log_likelihood(image, classParam):
    m=classParam[0]
    s2=classParam[1]
    #s=0 => p=1 => log(p)=0
    return -sum([math.log(2*math.pi*s2[i])/2
                + ((image[i]-m[i])**2)/s2[i]/2
                for i in range(len(image)) if s2[i]])

print('input', [i.shape for i in training_data])
param = learnML_all_parameters(training_data)
print('output', [(i[0].shape, i[1].shape) for i in param])

test_data = read_file("usps_test.txt")
print(log_likelihood(test_data[2][3], param[1]))

print([log_likelihood(test_data[0][0], param[i]) for i in range(10)])

def log_likelihoods(image, allParam):
    """"""
    return np.array([log_likelihood(image, i) for i in allParam])

print(log_likelihoods(test_data[1][5], param))

def classify_image(image, allParam):
    """"""
    return np.argmax(log_likelihoods(image,allParam))

print(classify_image(test_data[1][5], param))
print(classify_image(test_data[4][1], param))