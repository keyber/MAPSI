import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np

def loadData():
    with open('dataVelib.pkl','rb') as f:
        data = pkl.load(f)

    #data to matrix
    #3 coordonnées géographiques, Arrondissement, Places nStationales, Places dispo.
    return np.array([[s['alt'],
                      s['position']['lat'],
                      s['position']['lng'],
                      s['number']//1000,
                      s['bike_stands'],
                      s['available_bike_stands']]
                     #élimine les données incorrectes au passage
                     for s in data if 1<=s['number']//1000<=20])
def main():
    """INITIALISATION"""
    #latitude:x2, longitude:x1
    alt,x2,x1,arr,total,dispo = loadData().transpose()
    pos = np.array([x1,x2])
    del x1,x2
    nStation = alt.size
    stationPleine = np.array(dispo==0)#on ne peut pas y poser son vélo
    veloDispo = np.array(total - dispo >= 2)#on peut en prendre 2

    """CALCUL DES PROBAS"""
    pArr, intervArr,_ = plt.hist(arr, 20, density=True)
    nAlt = 30
    pAlt, intervAlt,_ = plt.hist(alt, nAlt, density=True)
    #plt.show()
    plt.close('all')

    indFromAlt = lambda x:\
        int((x - intervAlt[0]) * nAlt / (intervAlt[nAlt]-intervAlt[0]))

    #tableau des P(StationPleine|Altitude=a)
    pSP_alt = np.array([0]*nAlt)
    #compte le nombre de station pleine pour chaque alt
    for i in range(nStation):
        if stationPleine[i]:
            pSP_alt[indFromAlt(alt[i])] += 1
    #divise par le nombre de station de chaque alt
    #suppose les probas non nulles
    pSP_alt = pSP_alt / (pAlt * nStation)

    pVD_alt = np.array([0]*nAlt)
    for i in range(nStation):
        if veloDispo[i]:
            pVD_alt[indFromAlt(alt[i])] += 1
    pVD_alt = pVD_alt / (pAlt * nStation)

    pVD_arr = np.array([0]*20)
    for i in range(nStation):
        if veloDispo[i]:
            pVD_arr[int(arr[i]-1)] += 1
    pVD_arr = pVD_arr / (pArr * nStation)

    """AFFICHAGE STATIONS"""
    #fait varier les couleurs en premier
    style = np.array([(s,c) for s in "o^+*" for c in "rycmgbk"])

    plt.figure()
    for i in range(1,21):
        ind, = np.where(arr==i)
        color = style[i][1]
        plt.scatter(*pos[:,ind],marker=style[i-1][0],c=color,s=20, linewidths=0)
    
    plt.axis('equal')#affichage repère hortonormé
    plt.legend(range(1,21), fontsize=5)
    #plt.show()
    plt.close('all')

    #figure stations pleine/vide
    plt.figure()
    plt.axis('equal')
    #pas pratique pour combiner critère couleur avec d'autres critères
    plt.scatter(*pos[:,np.where(stationPleine)],c='r',s=16, linewidths=0)
    plt.scatter(*pos[:,np.where(veloDispo==0)],c='b',s=16, linewidths=0)
    plt.scatter(*pos[:,np.where((stationPleine | (veloDispo==0))==0)],c='g',s=16, linewidths=0)
    plt.legend(["plein", "vide", "autre"], fontsize=10)

    moy = np.mean(alt)
    med = np.median(alt)

    plt.figure()
    plt.axis('equal')
    #pas pratique pour la légende
    colors = np.array(['r' if stationPleine[i] else
                       'b' if not veloDispo[i] else 'g' for i in range(nStation)])
    ind, = np.where(alt > moy)
    plt.scatter(*pos[:,ind],c=colors[ind],s=16, linewidths=0)

    plt.figure()
    plt.axis('equal')
    ind, = np.where(alt > med)
    plt.scatter(*pos[:,ind],c=colors[ind],s=16, linewidths=0)

    plt.show()

main()



