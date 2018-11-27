import numpy as np
import pickle as pkl
from hmmlearn import hmm
import matplotlib.pyplot as plt

with open('genome_genes.pkl', 'rb') as f:
    data = pkl.load(f, encoding='latin1')

Xgenes  = data.get("genes") #Les genes, une array de arrays

Genome = data.get("genome") #le premier million de bp de Coli

Annotation = data.get("annotation") ##l'annotation sur le genome
##0 = non codant, 1 = gene sur le brin positif

### Quelques constantes
DNA = ["A", "C", "G", "T"]
stop_codons = ["TAA", "TAG", "TGA"]

#MODELE 1 (hmmlearn)

n_states_m1 = 4
# syntaxe objet python: créer un objet HMM
model1 = hmm.MultinomialHMM(n_components = n_states_m1)
a=1/200

tailleTot = np.sum([len(x) for x in Xgenes])
espTaille = tailleTot/len(Xgenes)/3#regroupe lettres 3 par 3
b=1/espTaille

Pi_m1 = np.array([1, 0, 0, 0]) ##on commence dans l'intergenique
A_m1 = np.array([[1-a, a  , 0, 0],
                 [0  , 0  , 1, 0],
                 [0  , 0  , 0, 1],
                 [b  , 1-b, 0, 0 ]])

Binter = np.array([.0,.0,.0,.0])
for i in Genome:
    Binter[i] += 1
Binter /= len(Genome)
Binter = [Binter]
print(Binter)

Bgene = np.zeros((3, n_states_m1))  #compte
for gene in Xgenes:
    for i in range(3, len(gene) - 3):  #sans premier ni dernier
        index = i % 3  #par paquet de 3
        Bgene[index][gene[i]] += 1
for line in Bgene:
    line /= np.sum(line)

print(Bgene)

# paramétrage de l'objet
model1.startprob_ = Pi_m1
model1.transmat_ = A_m1

B_m1 = np.vstack((Binter, Bgene))

model1.emissionprob_ = B_m1

vsbce, pred = model1.decode(np.reshape(Genome, (-1,1)), algorithm="viterbi")
#vsbce contient la log vsbce
#pred contient la sequence des etats predits (valeurs entieres entre 0 et 3)

#on peut regarder la proportion de positions bien predites
#en passant les etats codant a 1
sp = pred
sp[np.where(sp>=1)] = 1
percpred1 = float(np.sum(sp == Annotation))/len(Annotation)

print("precision modèle 1", percpred1)

#MODELE 2

startProba= np.array([.83,.14,.03])#A(C)GT
s=startProba*a#normalise pour sommer à "a"
A_m2 = np.array([
    #INTergenique
    #Début 0 1 2
    #Codon 0 1 2
    #Fin 0 1G 1A 2A 2G
    #(total = 12 (* 12))
    #int, d0  d1, d2, c0, c1,c2,f0,f1G,f1A,f2A,f2G
    [1-a, a,  0,   0,   0, 0, 0, 0,  0, 0, 0, 0],#inter devient inter ou D0
    [0  , 0,  1,   0,   0, 0, 0, 0, 0,  0, 0, 0],#d0 devient d1
    [0  , 0,  0,   1,   0, 0, 0, 0, 0,  0, 0, 0],#d1         d2
    [0  , 0,  0,   0,   1, 0, 0, 0, 0,  0, 0, 0],#d2         c0
    [0  , 0,  0,   0,   0, 1, 0, 0, 0,  0, 0, 0],#c0 devient c1
    [0  , 0,  0,   0,   0, 0, 1, 0, 0,  0, 0, 0],#c1 devient c2
    [0  , 0,  0,   0, 1-b, 0, 0, b, 0,  0, 0, 0],#c2 devient c0 ou f0
    [0  , 0,  0,   0,   0, 0, 0, 0,.5, .5, 0, 0],#f0 devient f1A ou f2G
    [0  , 0,  0,   0,   0, 0, 0, 0, 0,  0,.5,.5],#f1A devient f2A ou f2G
    [0  , 0,  0,   0,   0, 0, 0, 0, 0,  0, 0, 1],#f1G devient A
    [1  , 0,  0,   0,   0, 0, 0, 0, 0,  0, 0, 0],#f2A devient inter
    [1  , 0,  0,   0,   0, 0, 0, 0, 0,  0, 0, 0] #f2G devient inter
    ])

Bstart = np.array([[0.83, 0, 0.14, 0.03],
                   [0, 0, 0, 1],
                   [0, 0, 1, 0]])

#Codon stop : TAA, TAG, TGA
Bstop = np.array([[ 0, 0, 0, 1],#f0
                  [.5, 0, 0,.5],#f1A
                  [ 1, 0, 0, 0],#f2G
                  #"une seule lettre avec proba 1" :
                  [ 1, 0, 0, 0],#f1G
                  [ 1, 0, 0, 0],#f2A
                  ])

B_m2 = np.vstack((Binter, Bstart, Bgene, Bstop))
#print(B_m2)

Pi_m2 = [1,0,0,0,0,0,0,0,0,0,0,0]


model2 = hmm.MultinomialHMM(n_components = 12)
model2.startprob_ = Pi_m2
model2.transmat_ = A_m2

model2.emissionprob_ = B_m2

vsbce2, pred2 = model2.decode(np.reshape(Genome, (-1,1)), algorithm="viterbi")

#pred2, vsbce2 = viterbi(Genome,Pi_m2,A_m2,B_m2)

sp2 = pred2
sp2[np.where(sp2>=1)] = 1
percpred2 = float(np.sum(sp2 == Annotation) )/ len(Annotation)
print('précision modèle 2', percpred2)


r = range(6000)

"""#erreur 1, erreur 2, erreur 1&2, pas d'erreur
err1 = Annotation - pred
err2 = Annotation - pred2
colors = "yorg"
"""

plt.scatter(r,pred[r],       c='r', label='predictions 1')
plt.scatter(r,pred2[r],      c='g', label='predictions 2')
plt.scatter(r,Annotation[r], c='b', label='annotations')
plt.legend()
plt.show()
