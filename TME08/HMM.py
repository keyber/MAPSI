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

def distriCodons(Xgenes, n):
    dist = np.zeros((3,n))#compte
    for gene in Xgenes:
        for i in range(3, len(gene)-3):#sans premier ni dernier
            index = i % 3 #par paquet de 3
            dist[index][gene[i]] += 1
    return np.array([line/np.sum(line) for line in dist])

Bgene = distriCodons(Xgenes, 4)
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
percpred1 = float(np.sum(sp == Annotation) )/ len(Annotation)

print(percpred1)


predictions = pred[range(600)]
annotations = Annotation[range(600)]
plt.scatter(range(600),predictions, c='r', label='predictions')
plt.scatter(range(600),annotations, c='b', label='annotations')
plt.show()