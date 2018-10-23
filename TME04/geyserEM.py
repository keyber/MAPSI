import numpy as np
import math
import pylab
import matplotlib.pyplot as plt


def read_file(filename):
    """
    Lit le fichier contenant les données du geyser Old Faithful
    """
    # lecture de l'en-tête
    infile = open(filename, "r")
    for ligne in infile:
        if ligne.find("eruptions waiting") != -1:
            break
    
    # ici, on a la liste des temps d'éruption et des délais d'irruptions
    data = []
    for ligne in infile:
        nb_ligne, eruption, waiting = [float(x) for x in ligne.split()]
        data.append(eruption)
        data.append(waiting)
    infile.close()
    
    # transformation de la liste en tableau 2D
    data = np.asarray(data)
    data.shape = (int(data.size / 2), 2)
    
    return data


def normale2D(x, y, loi):
    mx, my, sx, sy, r = loi
    return 1 / (2 * math.pi * sx * sy * math.sqrt(1 - r * r)) \
           * math.exp(-1 / (2 * (1 - r * r)) *
                      (((x - mx) / sx) ** 2
                       - 2 * r * (x - mx) * (y - my) / (sx * sy)
                       + ((y - my) / sy) ** 2))


def test_dessine_1_normale(params):
    # récupération des paramètres
    mu_x, mu_z, sigma_x, sigma_z, rho = params
    
    # on détermine les coordonnées des coins de la figure
    x_min = mu_x - 2 * sigma_x
    x_max = mu_x + 2 * sigma_x
    z_min = mu_z - 2 * sigma_z
    z_max = mu_z + 2 * sigma_z
    
    # création de la grille
    x = np.linspace(x_min, x_max, 100)
    z = np.linspace(z_min, z_max, 100)
    X, Z = np.meshgrid(x, z)
    
    # calcul des normales
    norm = X.copy()
    for i in range(x.shape[0]):
        for j in range(z.shape[0]):
            norm[i, j] = normale2D(x[i], z[j], params)
    
    # affichage
    plt.figure()
    plt.contour(X, Z, norm)

    def test2D():
        for i in np.linspace(0, .9, 10):
            print(i)
            test_dessine_1_normale((0, 0, 5, 2, i))
        plt.show()


def dessine_normales(data, params, weights, bounds, ax):
    # on détermine les coordonnées des coins de la figure
    x_min = bounds[0]
    x_max = bounds[1]
    z_min = bounds[2]
    z_max = bounds[3]
    
    # création de la grille
    nb_x = nb_z = 100
    x = np.linspace(x_min, x_max, nb_x)
    z = np.linspace(z_min, z_max, nb_z)
    X, Z = np.meshgrid(x, z)
    
    # calcul des normales
    norm0 = np.zeros((nb_x, nb_z))
    for j in range(nb_z):
        for i in range(nb_x):
            norm0[j, i] = normale2D(x[i], z[j], params[0])  # * weights[0]
    norm1 = np.zeros((nb_x, nb_z))
    for j in range(nb_z):
        for i in range(nb_x):
            norm1[j, i] = normale2D(x[i], z[j], params[1])  # * weights[1]
    
    # affichages des normales et des points du dataset
    ax.contour(X, Z, norm0, cmap=pylab.cm.winter, alpha=0.5)
    ax.contour(X, Z, norm1, cmap=pylab.cm.autumn, alpha=0.5)
    for point in data:
        ax.plot(point[0], point[1], 'k+')


def find_bounds(data, params):
    # récupération des paramètres
    mu_x0, mu_z0, sigma_x0, sigma_z0, rho0 = params[0]
    mu_x1, mu_z1, sigma_x1, sigma_z1, rho1 = params[1]
    
    # calcul des coins
    x_min = min(mu_x0 - 2 * sigma_x0, mu_x1 - 2 * sigma_x1, data[:, 0].min())
    x_max = max(mu_x0 + 2 * sigma_x0, mu_x1 + 2 * sigma_x1, data[:, 0].max())
    z_min = min(mu_z0 - 2 * sigma_z0, mu_z1 - 2 * sigma_z1, data[:, 1].min())
    z_max = max(mu_z0 + 2 * sigma_z0, mu_z1 + 2 * sigma_z1, data[:, 1].max())
    
    return x_min, x_max, z_min, z_max

def Q_i(data, current_params, current_weights):
    """return Qi,t+1"""
    loi1,loi2 = current_params
    poids1, poids2 = current_weights
    a0 = np.array([normale2D(x,y, loi1) for x,y in data]) * poids1
    a1 = np.array([normale2D(x,y, loi2) for x,y in data]) * poids2
    q0 = a0 / (a0 + a1)
    q1 = a1 / (a0 + a1)
    return np.array([[q0[i], q1[i]] for i in range(q0.size)])

def M_step(data, Q):
    """return [loi1=m1m2s1s2r,loi2],[pi0,pi1]"""
    Q0 = Q[:, 0]
    Q1 = Q[:, 1]
    sumQ0 = Q0.sum()
    sumQ1 = Q1.sum()
    
    pi0 = sumQ0 / (sumQ0 + sumQ1)
    pi1 = sumQ1 / (sumQ0 + sumQ1)
    
    xi = data[:, 0]
    yi = data[:, 1]
    
    mux0 = (Q0 * xi).sum() / sumQ0
    mux1 = (Q1 * xi).sum() / sumQ1
    
    muy0 = (Q0 * yi).sum() / sumQ0
    muy1 = (Q1 * yi).sum() / sumQ1
    
    sigmax0 = math.sqrt((Q0 * (xi - mux0) ** 2).sum() / sumQ0)
    sigmax1 = math.sqrt((Q1 * (xi - mux1) ** 2).sum() / sumQ1)
    
    sigmay0 = math.sqrt((Q0 * (yi - muy0) ** 2).sum() / sumQ0)
    sigmay1 = math.sqrt((Q1 * (yi - muy1) ** 2).sum() / sumQ1)
    
    p0 = ((Q0 * ((xi - mux0) * (yi - muy0) / (sigmax0 * sigmay0))).sum()) / sumQ0
    p1 = ((Q1 * ((xi - mux1) * (yi - muy1) / (sigmax1 * sigmay1))).sum()) / sumQ1
    
    return (np.array([[mux0, muy0, sigmax0, sigmay0, p0], [mux1, muy1, sigmax1, sigmay1, p1]]),
            np.array([pi0, pi1]))

# calcul des bornes pour contenir toutes les lois normales calculées
def find_video_bounds ( data, res_EM ):
    bounds = np.asarray ( find_bounds ( data, res_EM[0][0] ) )
    for param in res_EM:
        new_bound = find_bounds ( data, param[0] )
        for i in [0,2]:
            bounds[i] = min ( bounds[i], new_bound[i] )
        for i in [1,3]:
            bounds[i] = max ( bounds[i], new_bound[i] )
    return bounds

def main():
    data = read_file("faithful.txt")

    # affichage des données : calcul des moyennes et variances des 2 colonnes
    mean1 = data[:, 0].mean()
    mean2 = data[:, 1].mean()
    std1 = data[:, 0].std()
    std2 = data[:, 1].std()
    
    print("moy1,2, std1,2",mean1,mean2,std1,std2)

    # les paramètres des 2 normales sont autour de ces moyennes
    params = np.array([(mean1 - 0.2, mean2 - 1, std1, std2, 0),
                       (mean1 + 0.2, mean2 + 1, std1, std2, 0)])
    weights = np.array([0.5, 0.5])
    
    res = [(params,weights)]
    for i in range(20):
        res.append((params,weights))
        Q = Q_i(data, params, weights)
        params,weights = M_step(data, Q)
        """affichage de la figure
        bounds = find_bounds(data, params)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        dessine_normales(data, params, weights, bounds, ax)
        plt.show()
        """

    bounds = find_video_bounds(data, res)
    import matplotlib.animation as animation

    # création de l'animation : tout d'abord on crée la figure qui sera animée
    fig = plt.figure()
    ax = fig.gca(xlim=(bounds[0], bounds[1]), ylim=(bounds[2], bounds[3]))

    # la fonction appelée à chaque pas de temps pour créer l'animation
    def animate(i):
        ax.cla()
        dessine_normales(data, res[i][0], res[i][1], bounds, ax)
        ax.text(5, 40, 'step = ' + str(i))
        print("step animate = %d" %i)

    # exécution de l'animation
    anim = animation.FuncAnimation(fig, animate, frames=len(res), interval=50)
    #plt.show()

    # éventuellement, sauver l'animation dans une vidéo
    Writer = animation.writers['html']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    anim.save('old_faithful.html', writer=writer)

main()
