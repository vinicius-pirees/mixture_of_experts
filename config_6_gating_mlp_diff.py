import numpy as np
from utils import softmax, dirac_delta

def mistura(Xtr, Ytr, m):
    Ntr = Xtr.shape[0]
    ne = Xtr.shape[1]
    ns = Ytr.shape[1]
    nh_gating = 5
    nh1 = 9
    nh2 = 7
    nh3 = 3
    nh4 = 4



    ##add bias
    Xtr = np.concatenate((Xtr, np.ones((Ntr, 1))), axis=1)
    ne = ne + 1

    ## Inicializa rede gating
    Wg = np.random.rand(nh_gating, ne)
    Wg2 = np.random.rand(m, nh_gating + 1)

    # Inicializa especialistas
    W1 = {}
    W2 = {}
    var = list(range(m))
    for i in range(2):
        W1[i] = np.random.rand(nh1, ne)
        W2[i] = np.random.rand(ns, nh1+1)
        var[i] = 1

    for i in range(2,4):
        W1[i] = np.random.rand(nh2, ne)
        W2[i] = np.random.rand(ns, nh2+1)
        var[i] = 1

    W1[4] = np.random.rand(nh3, ne)
    W2[4] = np.random.rand(ns, nh3 + 1)
    var[4] = 1

    W1[5] = np.random.rand(nh4, ne)
    W2[5] = np.random.rand(ns, nh4 + 1)
    var[5] = 1

    ##calcula saida
    Zg1 = np.dot(Xtr, Wg.T)
    ##tanh
    Ag1 = (np.exp(Zg1) - np.exp(-Zg1)) / (np.exp(Zg1) + np.exp(-Zg1))
    ##add bias
    Ag1 = np.concatenate((Ag1, np.ones((Ntr, 1))), axis=1)
    Zg2 = np.dot(Ag1, Wg2.T)
    Yg = softmax(Zg2)

    Ye = {}
    for i in range(m):
        Z1 = np.dot(Xtr, W1[i].T)
        A1 = (np.exp(Z1) - np.exp(-Z1)) / (np.exp(Z1)+np.exp(-Z1))
        ##add bias
        A1 = np.concatenate((A1, np.ones((Ntr, 1))), axis=1)
        Ye[i] = np.dot(A1, W2[i].T)
    Ym = np.zeros((Ntr, ns))
    for i in range(m):
        Yge = Yg[:, i].reshape(Ntr, 1)
        Ym = Ym + Ye[i] * Yge

    ##calculo da verossimilhanca
    Py = np.zeros((Ntr, m))
    for i in range(m):
        Yaux = Ye[i]
        for j in range(Ntr):
            diff = Ytr[j, :] - Yaux[j, :]
            Py[j, i] = np.exp(np.dot(-diff, diff.T) / (2 * var[i]))

    likelihood = np.sum(np.log(np.sum(Yg * Py, axis=1, keepdims=True)))
    likelihood_ant = 0
    nit = 0
    nitmax = 20

    while np.abs(likelihood - likelihood_ant) > 1e-3 and nit < nitmax:
        nit = nit + 1
        # Passo E
        haux = Yg * Py
        h = haux / np.dot(np.sum(haux, axis=1, keepdims=True), np.ones((1, m)))
        ##Passo M
        Wg, Wg2 = maximiza_gating(Wg, Wg2, Xtr, m, h)
        for i in range(m):
            W1[i], W2[i], var[i] = maximiza_expert(W1[i], W2[i],var[i], Xtr, Ytr, h[:, i].reshape(Ntr, 1))
        likelihood_ant = likelihood

        ##calcula saida
        Zg1 = np.dot(Xtr, Wg.T)
        ##tanh
        Ag1 = (np.exp(Zg1) - np.exp(-Zg1)) / (np.exp(Zg1) + np.exp(-Zg1))
        ##add bias
        Ag1 = np.concatenate((Ag1, np.ones((Ntr, 1))), axis=1)
        Zg2 = np.dot(Ag1, Wg2.T)
        Yg = softmax(Zg2)

        Ye = {}
        for i in range(m):
            Z1 = np.dot(Xtr, W1[i].T)
            A1 = (np.exp(Z1) - np.exp(-Z1)) / (np.exp(Z1) + np.exp(-Z1))
            ##add bias
            A1 = np.concatenate((A1, np.ones((Ntr, 1))), axis=1)
            Ye[i] = np.dot(A1, W2[i].T)
        Ym = np.zeros((Ntr, ns))
        for i in range(m):
            Yge = Yg[:, i].reshape(Ntr, 1)
            Ym = Ym + Ye[i] * Yge

        ##calculo da funcao de verossimilhanca
        Py = np.zeros((Ntr, m))
        for i in range(m):
            Yaux = Ye[i]
            for j in range(Ntr):
                diff = Ytr[j, :] - Yaux[j, :]
                Py[j, i] = np.exp(np.dot(-diff, diff.T) / (2 * var[i]))

        likelihood = np.sum(np.log(np.sum(Yg * Py, axis=1, keepdims=True)))

    errov = Ym - Ytr
    EQMv = 1 / Ntr * np.sum(errov * errov)


    me = {}
    me['gating_Wg'] = Wg
    me['gating_Wg2'] = Wg2
    me['expert_W1'] = W1
    me['expert_W2'] = W2
    me['expert_var'] = var
    me['last_error'] = EQMv
    me['last_likelihood'] = likelihood

    return me


def maximiza_gating(Wg, Wg2, Xtr, m, h):
    N = Xtr.shape[0]
    nh = Wg2.shape[1] - 1
    ns = m


    Z1 = np.dot(Xtr, Wg.T)
    ##tanh
    A1 = (np.exp(Z1) - np.exp(-Z1)) / (np.exp(Z1) + np.exp(-Z1))
    ##add bias
    A1 = np.concatenate((A1, np.ones((N, 1))), axis=1)
    Z2 = np.dot(A1, Wg2.T)
    Yg = softmax(Z2)


    dC_Z2 = np.zeros((N, ns))

    for i in range(0, ns):
        z2 = 0
        for j in range(0, ns):
            dC_dAj = (h - Yg)[:, [j]]
            dAj_dZi = (Yg[:, [i]] * (dirac_delta(i, j) - Yg[:, [j]]))
            z2 = dC_dAj * dAj_dZi
        dC_Z2[:, [i]] = z2

    dWg2 = 1 / N * np.dot((dC_Z2).T, A1)
    dWg = 1 / N * np.dot((np.dot(dC_Z2, Wg2[:, :nh]) * (1 - (A1[:, :nh] * A1[:, :nh]))).T, Xtr)


    nit = 0
    nitmax = 50000
    alfa = 0.1

    while np.linalg.norm(dWg2) > 1e-5 and nit < nitmax:
        nit = nit + 1
        Wg = Wg + (alfa * dWg)
        Wg2 = Wg2 + (alfa * dWg2)

        Z1 = np.dot(Xtr, Wg.T)
        ##tanh
        A1 = (np.exp(Z1) - np.exp(-Z1)) / (np.exp(Z1) + np.exp(-Z1))
        ##add bias
        A1 = np.concatenate((A1, np.ones((N, 1))), axis=1)
        Z2 = np.dot(A1, Wg2.T)
        Yg = softmax(Z2)

        ns = Yg.shape[1]

        dC_Z2 = np.zeros((N, ns))

        for i in range(0, ns):
            z2 = 0
            for j in range(0, ns):
                dC_dAj = (h - Yg)[:, [j]]
                dAj_dZi = (Yg[:, [i]] * (dirac_delta(i, j) - Yg[:, [j]]))
                z2 = dC_dAj * dAj_dZi
            dC_Z2[:, [i]] = z2

        dWg2 = 1 / N * np.dot((dC_Z2).T, A1)
        dWg = 1 / N * np.dot((np.dot(dC_Z2, Wg2[:, :nh]) * (1 - (A1[:, :nh] * A1[:, :nh]))).T, Xtr)

    return Wg,Wg2


def maximiza_expert(W1, W2, var, Xtr, Ytr, h):


    Z1 = np.dot(Xtr, W1.T)
    A1 = (np.exp(Z1) - np.exp(-Z1)) / (np.exp(Z1)+np.exp(-Z1))
    ##add bias
    A1 = np.concatenate((A1, np.ones((Ytr.shape[0], 1))), axis=1)
    Ye = np.dot(A1, W2.T)

    N = Ye.shape[0]
    ns = Ye.shape[1]
    nh = W2.shape[1] - 1

    dC_dZ2 = ((h / var) * (Ytr - Ye))


    dW2 = 1/N * np.dot(dC_dZ2.T, A1)
    dW1 = 1/N * np.dot((np.dot(dC_dZ2, W2[:,:nh]) * (1-(A1[:,:nh] * A1[:,:nh]))).T , Xtr)

    nit = 0
    nitmax = 50000
    alfa = 0.1

    while np.linalg.norm(dW2) > 1e-5 and nit < nitmax:
        nit = nit + 1
        W1 = W1 + (alfa * dW1)
        W2 = W2 + (alfa * dW2)

        Z1 = np.dot(Xtr, W1.T)
        A1 = (np.exp(Z1) - np.exp(-Z1)) / (np.exp(Z1) + np.exp(-Z1))
        ##add bias
        A1 = np.concatenate((A1, np.ones((N, 1))), axis=1)
        Ye = np.dot(A1, W2.T)

        dC_dZ2 = ((h / var) * (Ytr - Ye))

        dW2 = 1 / N * np.dot(dC_dZ2.T, A1)
        dW1 = 1 / N * np.dot((np.dot(dC_dZ2, W2[:, :nh]) * (1 - (A1[:, :nh] * A1[:, :nh]))).T, Xtr)

    diff = Ytr - Ye
    soma = 0

    for i in range(N):
        soma = soma + (h[i] * np.dot(diff[i, :], diff[i, :].T))[0]
    var = max(0.05, (1 / ns) * soma / np.sum(h))

    return W1,W2, var