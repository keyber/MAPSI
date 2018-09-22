import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np

f= open('dataVelib.pkl','rb')
data = pkl.load(f)
f.close()

#data to matrix
#3 coordonnées géographiques, Arrondissement, Places totales, Places dispo.
altInd = 0
arrInd = 3
totInd = 4
dispoInd=5
data = np.array([[s['alt'],
    s['position']['lat'],
    s['position']['lng'],
    s['number']//1000,
    s['bike_stands'],
    s['available_bike_stands']]
    #élimine les données incorrectes au passage
   for s in data if 1<=s['number']//1000<=20])

tot = len(data)

pArr, intervalArr = plt.hist(data[:,arrInd]/tot, 20)
n = 100
pAlt, intervalAlt = plt.hist(data[:,altInd]/tot, n)
indFromAlt = lambda x:\
    (x - intervalAlt[0]) * n / (intervalAlt[n+1]-intervalAlt[0])

sP = np.array(data[:,dispoInd]==0)
#tableau des P(StationPleine|Altitude=a)
pSP_alt = np.array([0]*n)
for i in range(tot):
    if sP[i]:
        pSP_alt[indFromAlt(i)] += 1
#normalise
pSP_alt = pSP_alt / (pAlt * tot)


