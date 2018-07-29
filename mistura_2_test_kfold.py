import numpy as np
import pandas
import os
from mistura_2 import mistura
from k_fold_validation import k_fold
from utils import softmax, series_to_supervised

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%  Mistura de Especialista  %%%%%%%%%%%%%%%%%%%%%%
# %%%%%Rede Especialista - Perceptron%%%%%%%%%%%%%%%%%%%
# %%%%%Rede Gating - Perceptron%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# Xtr - entrada de treinamento
# Ytr - saida de treinamento
# Wg - rede gating
# W - especialistas

filename = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname('treinamento.txt'))) + '/treinamento-1.txt'
series = pandas.read_csv(filename,  header=None)

if __name__ == "__main__":

    D = series_to_supervised(series, 21).values
    k = 4

    folded_partitions = k_fold(D, k)

    for partition in folded_partitions:
        Dtr = folded_partitions[partition][0]
        Dv = folded_partitions[partition][1]

        Xtr = Dtr[:, 0:-1]
        Ytr = Dtr[:, -1].reshape(Xtr.shape[0], 1)

        Xv = Dv[:, 0:-1]
        Yv = Dv[:, -1].reshape(Xv.shape[0], 1)

        m = 4
        hidden_units = 7

        me = mistura(Xtr, Ytr, m, hidden_units)

        ## Teste

        Nv = Xv.shape[0]
        ne = Xv.shape[1]
        ns = Yv.shape[1]

        ##add bias
        Xv = np.concatenate((Xv, np.ones((Nv, 1))), axis=1)
        ne = ne + 1

        Wg = me['gating']
        W1 = me['expert_W1']
        W2 = me['expert_W2']
        var = me['expert_var']

        ##calcula saida
        Yg = softmax(np.dot(Xv, Wg.T))
        Ye = {}
        for i in range(m):
            Z1 = np.dot(Xv, W1[i].T)
            A1 = (np.exp(Z1) - np.exp(-Z1)) / (np.exp(Z1) + np.exp(-Z1))
            ##add bias
            A1 = np.concatenate((A1, np.ones((Nv, 1))), axis=1)
            Ye[i] = np.dot(A1, W2[i].T)
        Ym = np.zeros((Nv, ns))
        for i in range(m):
            Yge = Yg[:, i].reshape(Nv, 1)
            Ym = Ym + Ye[i] * Yge

        ##calculo da funcao de verossimilhanca
        Py = np.zeros((Nv, m))
        for i in range(m):
            Yaux = Ye[i]
            for j in range(Nv):
                diff = Yv[j, :] - Yaux[j, :]
                Py[j, i] = np.exp(np.dot(-diff, diff.T) / (2 * var[i]))

        likelihood = np.sum(np.log(np.sum(Yg * Py, axis=1, keepdims=True)))

        erro_medio = np.sqrt(np.square(np.sum(Yv - Ym))/Nv)

        print("Resultados fold", partition)
        print (likelihood)

        errov = Ym - Yv
        EQMv = 1 / Nv * np.sum(errov * errov)

        print(EQMv)

        print('\n', me, '\n')







