import numpy as np

def calc_saida(X,Y,W):
    A = np.dot(X,W.T)
    erro = A - Y
    return A, erro


def grad(X,Y,W,N):
    A,erro = calc_saida(X,Y,W)
    dW = 1/N * np.dot(erro.T, X)
    return dW


def perceptron(maxepocas, alfa, X, Y):
    N = Y.shape[0]
    ns = Y.shape[1]
    ##add bias
    X = np.concatenate((X, np.ones((N, 1))), axis=1)

    ne = X.shape[1]
    W = np.random.rand(ns, ne) / 5

    ##Calcula saida
    A, erro = calc_saida(X, Y, W)
    EQM = 1 / N * np.sum(erro * erro)

    vet_erro = np.array([EQM])
    new_vet = vet_erro

    nepocas = 0

    while nepocas < maxepocas:
        nepocas = nepocas + 1
        dW = grad(X, Y, W, N);
        W = W - alfa * dW
        A, erro = calc_saida(X, Y, W)
        EQM = 1 / N * np.sum(erro * erro)
        new_vet = np.append(new_vet, EQM)

    return W, new_vet






