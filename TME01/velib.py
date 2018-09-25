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

def correlation(valsA, valsB, pab):
    """:param pab loi jointe de a et b
    :param valsA valeurs prises par la variable aléatoire A
    :param valsB valeurs prises par la variable aléatoire B"""
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
    #print(moyA, stdA, pa)
    #print(moyB, stdB, pb)

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
    """ecart type de A et B doivent être non nuls"""
    assert len(valsA)==len(valsB)
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

def correlationPVar(pA_B, pB, varB):
    """P[i] = [1 avec proba p[i], 0 avec proba 1-p[i]] <-> varB[i] = p[i]. dem:
    p(Alt=alt, VD=vd) = p(VD=d|Alt=a)*p(Alt=a)
     la covariance est la somme sur a de
         P   * p(a) * (a-am) * (1-pm)
     + (1-P) * p(a) * (a-am) * (0-pm)
     = p(a)*(a-am) * [(1-pm)*P - dm*(1-P)]
     = p(a)*(a-am) * (P - pmP - pm + pmP)
     = p(a)*(a-am) * (P - pm)"""
    assert pA_B.ndim==1 and pA_B.shape==pB.shape and pB.shape==varB.shape
    pab = np.array([pA_B*pB, [(1 - pA_B[i]) * pB[i] for i in range(len(pA_B))]])
    return correlation(np.array([1,0]), varB, pab)

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
    alt,x2,x1,arr,total,nbPlace = loadData().transpose()
    arr = arr.astype(int)
    pos = np.array([x1,x2])
    del x1,x2
    nStation = len(alt)
    stationPleine = np.array(nbPlace==0)#on ne peut pas y poser son vélo
    veloDispo = np.array(total - nbPlace >= 2)#on peut en prendre 2

    """CALCUL DES PROBAS"""
    #interv[-1] est le maximum et est atteint
    pArr,_,_ = plt.hist(arr, 20)#density=True marche pas comme on le souhaiterait
    pArr = np.array(pArr)/nStation
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
    print("variance, corrélation disponibilité - altitude", correlationBrute(alt, veloDispo))

    correlation_vd_alt = correlationPVar(pVD_alt, pAlt, np.array(intervAlt[:-1]))
    print("variance, corrélation disponibilité - regoupement altitude", correlation_vd_alt)


    print("variance, corrélation disponibilité - arrondissement", correlationBrute(arr, veloDispo))

    #tri des arrondissements par proba vd croissante
    sortedIndexes = np.argsort(np.vectorize(lambda x:pVD_arr[x-1]+x/10000)(arr))
    arrS = arr[sortedIndexes]
    num_actuel = 0#pas dans arr
    ind_actuel = 0
    arrToInd={}
    for i in range(nStation):
        if arrS[i]!=num_actuel:
            num_actuel=arrS[i]
            ind_actuel+=1
            arrToInd[num_actuel]=ind_actuel
        arrS[i]=ind_actuel

    print("variance, corrélation disponibilité - arrondissement triés par taux de disponibilité croissante",
          correlationBrute(arrS,veloDispo[sortedIndexes]))

    correlation_vd_arr = correlationPVar(pVD_arr, pArr, np.array([arrToInd[i] for i in range(1,21)]))
    print("variance, corrélation disponibilité - arrondissement triés par taux de disponibilité croissante bis",
        correlation_vd_arr)

main()



