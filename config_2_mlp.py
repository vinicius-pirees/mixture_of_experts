import numpy as np


def calc_saida(X, Y, W1, W2, N):
    Z1 = np.dot(X, W1.T)
    ##tanh
    A1 = (np.exp(Z1) - np.exp(-Z1)) / (np.exp(Z1) + np.exp(-Z1))
    ##add bias
    A1 = np.concatenate((A1, np.ones((N, 1))), axis=1)

    Z2 = np.dot(A1, W2.T)
    A2 = Z2

    erro = A2 - Y

    return A1, A2, erro


def grad(X, Y, W1, W2, N):
    A1, A2, erro = calc_saida(X, Y, W1, W2, N)

    nh = W2.shape[1] - 1
    ns = Y.shape[1]

    dC_dAj = erro
    dAj_dZi = 1

    dC_Z2 = dC_dAj * dAj_dZi

    dW2 = 1 / N * np.dot(dC_Z2.T, A1)
    dW1 = 1 / N * np.dot((np.dot(dC_Z2, W2[:, :nh]) * (1 - (A1[:, :nh] * A1[:, :nh]))).T, X)
    return dW2, dW1


def perceptron(X, Y, alfa, nepocasmax, hidden_units):
    N = Y.shape[0]
    ns = Y.shape[1]
    nh = hidden_units

    ##add bias
    X = np.concatenate((X, np.ones((N, 1))), axis=1)

    ne = X.shape[1]
    W1 = np.random.rand(nh, ne) / 5
    W2 = np.random.rand(ns, nh + 1) / 5

    ##Calcula saida
    A1, A2, erro = calc_saida(X, Y, W1, W2, N)
    EQM = 1 / N * np.sum(erro * erro)

    vet_erro = np.array([EQM])
    new_vet = vet_erro
    nepocas = 0


    while EQM > 1e-4 and nepocas < nepocasmax:
        nepocas = nepocas + 1
        dW2, dW1 = grad(X, Y, W1, W2, N)

        W1 = W1 - (alfa * dW1)
        W2 = W2 - (alfa * dW2)

        A1, A2, erro = calc_saida(X, Y, W1, W2, N)
        EQM = 1 / N * np.sum(erro * erro)
        new_vet = np.append(new_vet, EQM)

    return W1, W2, new_vet




