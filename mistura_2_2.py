import numpy as np
from utils import softmax, dirac_delta

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%  Mistura de Especialista  %%%%%%%%%%%%%%%%%%%%%%
# %%%%%Rede Especialista - Perceptron%%%%%%%%%%%%%%%%%%%
# %%%%%Rede Gating - Perceptron%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# Xtr - entrada de treinamento
# Ytr - saida de treinamento
# Wg - rede gating
# W - especialistas



def mistura(Xtr, Ytr, m):
    Ntr = Xtr.shape[0]
    ne = Xtr.shape[1]
    ns = Ytr.shape[1]
    nh1 = 4
    nh2 = 3
    nh3 = 6
    nh4 = 9



    ##add bias
    Xtr = np.concatenate((Xtr, np.ones((Ntr, 1))), axis=1)
    ne = ne + 1

    ## Inicializa rede gating
    Wg = np.random.rand(m, ne)

    # Inicializa especialistas
    W1 = {}
    W2 = {}
    var = list(range(m))
    W1[0] = np.random.rand(nh1, ne)
    W2[0] = np.random.rand(ns, nh1+1)
    var[0] = 1

    W1[1] = np.random.rand(nh2, ne)
    W2[1] = np.random.rand(ns, nh2+1)
    var[1] = 1

    W1[2] = np.random.rand(nh3, ne)
    W2[2] = np.random.rand(ns, nh3+1)
    var[2] = 1

    W1[3] = np.random.rand(nh4, ne)
    W2[3] = np.random.rand(ns, nh4+1)
    var[3] = 1

    W1[4] = np.random.rand(ns, ne)
    var[4] = 1

    W1[5] = np.random.rand(ns, ne)
    var[5] = 1

    ##calcula saida
    Yg = softmax(np.dot(Xtr, Wg.T))

    Ye = {}
    for i in range(4):
        Z1 = np.dot(Xtr, W1[i].T)
        A1 = (np.exp(Z1) - np.exp(-Z1)) / (np.exp(Z1)+np.exp(-Z1))
        ##add bias
        A1 = np.concatenate((A1, np.ones((Ntr, 1))), axis=1)
        Ye[i] = np.dot(A1, W2[i].T)

    for i in range(4,6):
        Ye[i] = np.dot(Xtr, W1[i].T)


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
        Wg = maximiza_gating(Wg, Xtr, m, h)

        W1[0], W2[0], var[0] = maximiza_expert_1(W1[0], W2[0],var[0], Xtr, Ytr, h[:, 0].reshape(Ntr, 1))
        W1[1], W2[1], var[1] = maximiza_expert_2(W1[1], W2[1], var[1], Xtr, Ytr, h[:, 1].reshape(Ntr, 1))
        W1[2], W2[2], var[2] = maximiza_expert_3(W1[2], W2[2], var[2], Xtr, Ytr, h[:, 2].reshape(Ntr, 1))
        W1[3], W2[3], var[3] = maximiza_expert_4(W1[3], W2[3], var[3], Xtr, Ytr, h[:, 3].reshape(Ntr, 1))
        W1[4], var[4] = maximiza_expert_linear_1(W1[4], var[4], Xtr, Ytr, h[:, 4].reshape(Ntr, 1))
        W1[5], var[5] = maximiza_expert_linear_2(W1[5], var[5], Xtr, Ytr, h[:, 5].reshape(Ntr, 1))

        likelihood_ant = likelihood

        ##calcula saida
        Yg = softmax(np.dot(Xtr, Wg.T))

        Ye = {}
        for i in range(4):
            Z1 = np.dot(Xtr, W1[i].T)
            A1 = (np.exp(Z1) - np.exp(-Z1)) / (np.exp(Z1) + np.exp(-Z1))
            ##add bias
            A1 = np.concatenate((A1, np.ones((Ntr, 1))), axis=1)
            Ye[i] = np.dot(A1, W2[i].T)

        for i in range(4,6):
            Ye[i] = np.dot(Xtr, W1[i].T)

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

    me = {}
    me['gating'] = Wg
    me['expert_W1'] = W1
    me['expert_W2'] = W2
    me['expert_var'] = var

    return me


def maximiza_gating(Wg, Xtr, m, h):
    N = Xtr.shape[0]
    ne = Xtr.shape[1]
    ns = m

    Yg = softmax(np.dot(Xtr, Wg.T))

    dW = np.zeros((ns, ne))

    for i in range(0, ns):
        w = 0
        for j in range(0, ns):
            dC_dAj = (h - Yg)[:, [j]]
            dAj_dZi = (Yg[:, [i]] * (dirac_delta(i, j) - Yg[:, [j]]))
            dZi_dWi = Xtr
            w += np.dot((dC_dAj * dAj_dZi).T, dZi_dWi)
        dW[[i], :] = w

    dW = 1 / N * dW

    nit = 0
    nitmax = 10000
    alfa = 0.1



    while np.linalg.norm(dW) > 1e-5 and nit < nitmax:
        nit = nit + 1
        Wg = Wg + (alfa * dW)
        Yg = softmax(np.dot(Xtr, Wg.T))

        dW = np.zeros((ns, ne))

        for i in range(0, ns):
            w = 0
            for j in range(0, ns):
                dC_dAj = (h - Yg)[:, [j]]
                dAj_dZi = (Yg[:, [i]] * (dirac_delta(i, j) - Yg[:, [j]]))
                dZi_dWi = Xtr
                w += np.dot((dC_dAj * dAj_dZi).T, dZi_dWi)
            dW[[i], :] = w

        dW = 1 / N * dW

    return Wg


def maximiza_expert_1(W1, W2, var, Xtr, Ytr, h):

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
    nitmax = 30000
    alfa = 0.12

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


def maximiza_expert_2(W1, W2, var, Xtr, Ytr, h):

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
    nitmax = 20000
    alfa = 0.09

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


def maximiza_expert_3(W1, W2, var, Xtr, Ytr, h):

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
    nitmax = 8000
    alfa = 0.3

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



def maximiza_expert_4(W1, W2, var, Xtr, Ytr, h):

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
    nitmax = 7000
    alfa = 0.4

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



def maximiza_expert_linear_1(W, var, Xtr, Ytr, h):
    Ye = np.dot(Xtr, W.T)
    N = Ye.shape[0]
    ns = Ye.shape[1]

    grad = np.dot(((h / var) * (Ytr - Ye)).T, Xtr / N)

    dir = grad
    nit = 0
    nitmax = 10000
    alfa = 0.19

    while np.linalg.norm(grad) > 1e-5 and nit < nitmax:
        nit = nit + 1
        W = W + (alfa * dir)
        Ye = np.dot(Xtr, W.T)
        grad = np.dot(((h / var) * (Ytr - Ye)).T, Xtr / N)
        dir = grad

    diff = Ytr - Ye
    soma = 0

    for i in range(N):
        soma = soma + (h[i] * np.dot(diff[i, :], diff[i, :].T))[0]
    var = max(0.05, (1 / ns) * soma / np.sum(h))

    return W, var

def maximiza_expert_linear_2(W, var, Xtr, Ytr, h):
    Ye = np.dot(Xtr, W.T)
    N = Ye.shape[0]
    ns = Ye.shape[1]

    grad = np.dot(((h / var) * (Ytr - Ye)).T, Xtr / N)

    dir = grad
    nit = 0
    nitmax = 20000
    alfa = 0.3

    while np.linalg.norm(grad) > 1e-5 and nit < nitmax:
        nit = nit + 1
        W = W + (alfa * dir)
        Ye = np.dot(Xtr, W.T)
        grad = np.dot(((h / var) * (Ytr - Ye)).T, Xtr / N)
        dir = grad

    diff = Ytr - Ye
    soma = 0

    for i in range(N):
        soma = soma + (h[i] * np.dot(diff[i, :], diff[i, :].T))[0]
    var = max(0.05, (1 / ns) * soma / np.sum(h))

    return W, var




