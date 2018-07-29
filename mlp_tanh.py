import numpy as np
import pandas
import os
from utils import series_to_supervised
from k_fold_validation import k_fold

filename = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname('treinamento.txt'))) + '/treinamento-1.txt'
series = pandas.read_csv(filename,  header=None)


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


if __name__ == "__main__":

    D = series_to_supervised(series, 5).values
    k = 5

    folded_partitions = k_fold(D, k)

    sum_errors = 0

    for partition in folded_partitions:
        Dtr = folded_partitions[partition][0]
        Dv = folded_partitions[partition][1]

        Xtr = Dtr[:, 0:-1]
        Ytr = Dtr[:, -1].reshape(Xtr.shape[0], 1)

        Xv = Dv[:, 0:-1]
        Yv = Dv[:, -1].reshape(Xv.shape[0], 1)

        hidden_units = 7

        W1,W2,vet_erro=perceptron(Xtr,Ytr,0.1,50000,hidden_units)

        Nv = Yv.shape[0]

        ##add bias
        Xv = np.concatenate((Xv, np.ones((Nv,1))), axis=1)

        ##Calcula saida
        A1v,A2v,errov = calc_saida(Xv,Yv,W1,W2,Nv)
        EQMv = 1/Nv * np.sum(errov*errov)

        sum_errors += EQMv

        print("Resultados fold", partition)
        print("Erro Teste", EQMv)
        print("W1",W1,"W2",W2)

    print("Erro geral", sum_errors/k)
