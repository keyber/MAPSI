import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import math

def assertDensity(a):
    if abs(np.sum(a)-1)>.001 or not areProba(a):
        print(a.shape, a.size)
        print(a)
        print(np.sum(a))
        assert False
def areProba(a):
    return not np.any((a<0)|(a>1))

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
def correlation(valsA, valsB, pab):
    """ecart type de A et B non nuls"""
    assert (*valsA.shape, *valsB.shape) == pab.shape
    assert areProba(pab)
    pa = np.array([np.sum(pab[i,:]) for i in range(len(valsA))])
    assertDensity(pa)
    #espérance: somme des x * P(x)
    moyA = np.sum(pa*valsA)
    #variance: somme des P(x) * (x-moyA)**2
    stdA = math.sqrt(sum([pa[i] * (valsA[i] - moyA) **2 for i in range(len(valsA))]))
    pb = np.array([np.sum(pab[:,i]) for i in range(len(valsB))])
    assertDensity(pb)
    moyB = np.sum(pb*valsB)
    stdB = math.sqrt(sum([pb[i] * (valsB[i] - moyB) **2 for i in range(len(valsB))]))

    print(moyA, stdA, pa)
    print(moyB, stdB, pb)

    #covariance = somme des p * (x-E(x))(y-E(y))
    covariance = 0
    for i in range(len(valsA)):
        for j in range(len(valsB)):
            covariance += pab[i][j] * (valsA[i] - moyA) * (valsB[j] - moyB)

    #coef corrélation = covariance / ecarttype1*ecarttype2
    correl = covariance / (stdA*stdB)
    assert abs(correl)<=1.001
    return covariance, correl

def correlationBrute(valsA, valsB):
    """ecart type de A et B non nuls"""
    assert(len(valsA)==len(valsB))
    n = len(valsA)
    moyA = np.mean(valsA)
    stdA = np.std(valsA)
    moyB = np.mean(valsB)
    stdB = np.std(valsB)
    covariance = 0
    for i in range(n):
        covariance += 1/n * (valsA[i] - moyA) * (valsB[i] - moyB)
    correl = covariance / (stdA*stdB)
    assert abs(correl)<=1.001
    return covariance, correl

def main():
    """INITIALISATION"""
    #latitude:x2, longitude:x1
    alt,x2,x1,arr,total,nbPlace = loadData().transpose()
    pos = np.array([x1,x2])
    del x1,x2
    nStation = len(alt)
    stationPleine = np.array(nbPlace==0)#on ne peut pas y poser son vélo
    veloDispo = np.array(total - nbPlace >= 2)#on peut en prendre 2

    """CALCUL DES PROBAS"""
    #interv[-1] est le maximum et est atteint
    pArr, intervArr,_ = plt.hist(arr,20)#density=True ne donne pas exactement le résutat attendu
    pArr = np.array(pArr/nStation)
    nAlt = 30
    pAlt, intervAlt,_ = plt.hist(alt, nAlt)
    pAlt = np.array(pAlt/nStation)
    #plt.show()
    plt.close('all')
    assertDensity(pArr)
    assertDensity(pAlt)

    #vmax arrive dans un autre intervale, on retourne celui d'avant
    indFromAlt = lambda x:\
        min(nAlt-1, int((x - intervAlt[0]) * nAlt / (intervAlt[nAlt]-intervAlt[0])))

    #normalement égal à pAlt*nStation
    nStation_alt = np.array([0]*nAlt)
    for i in range(nStation):
        nStation_alt[indFromAlt(alt[i])] += 1

    #tableau des P(StationPleine|Altitude=a)
    pSP_alt = np.array([0]*nAlt)
    #compte le nombre de station pleine pour chaque alt
    for i in range(nStation):
        if stationPleine[i]:
            pSP_alt[indFromAlt(alt[i])] += 1
    #divise par le nombre de station de chaque alt
    #suppose les probas non nulles
    pSP_alt = pSP_alt / nStation_alt
    assert areProba(pSP_alt)

    pVD_alt = np.array([0]*nAlt)
    for i in range(nStation):
        if veloDispo[i]:
            pVD_alt[indFromAlt(alt[i])] += 1
    pVD_alt = pVD_alt / (pAlt * nStation)
    assert areProba(pVD_alt)

    pVD_arr = np.array([0]*20)
    for i in range(nStation):
        if veloDispo[i]:
            pVD_arr[int(arr[i]-1)] += 1
    pVD_arr = pVD_arr / (pArr * nStation)
    assert areProba(pVD_arr)

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

    altMoy = np.mean(alt)
    altMed = np.median(alt)

    plt.figure()
    plt.axis('equal')
    #pas pratique pour la légende
    colors = np.array(['r' if stationPleine[i] else
                       'b' if not veloDispo[i] else 'g' for i in range(nStation)])
    ind, = np.where(alt > altMoy)
    plt.scatter(*pos[:,ind],c=colors[ind],s=16, linewidths=0)

    plt.figure()
    plt.axis('equal')
    ind, = np.where(alt > altMed)
    plt.scatter(*pos[:,ind],c=colors[ind],s=16, linewidths=0)

    #plt.show()

    """CORRELATION"""
    print("corrélation directe", correlationBrute(alt, veloDispo))

    pab = np.array([pVD_alt[:]*pAlt, [(1-pVD_alt[i])*pAlt[i] for i in range(len(pVD_alt))]])

    correlation_vd_alt = correlation(np.array([1,0]), intervAlt[:-1], pab)
    print("corrélation générique regoupement altitude", correlation_vd_alt)

    #VD, VD_alt 1 avec proba vd[i], 0 avec proba 1-vd[i]
    #revient à faire vd[i] avec proba de 1
    """démonstration
    p(Alt=alt, VD=vd) = p(VD=d|Alt=a)*p(Alt=a)
     la covariance est la somme sur a de
          VD_alt[a]  * p(a) * (a-am) * (1-dm)
     + (1-VD_alt[a]) * p(a) * (a-am) * (0-dm)
     = p(a)*(a-am) * [(1-dm)*VD_alt[a] - dm*(1-VD_alt[a])]
     = p(a)*(a-am) * (VD - dmVD - dm + dmVD)
     = p(a)*(a-am) * (VD_alt[a] - dm)"""
    covariance_vd_alt = 0
    for i in range(len(pAlt)):
        covariance_vd_alt += pAlt[i] * (intervAlt[i] - np.mean(intervAlt)) * (pVD_alt[i] - np.mean(pVD_alt))
    correlation_vd_alt = covariance_vd_alt / (np.std(intervAlt) * np.std(pVD_alt))
    print("correlation proba-variable regroupement altitude" ,(covariance_vd_alt, correlation_vd_alt))

    #tri des arrondissements par proba vd croissante

main()



