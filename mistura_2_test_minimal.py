import numpy as np
import pandas
import os
from mistura_2 import mistura, softmax




if __name__ == "__main__":

    Xtr = np.array([[0.5348242,0.4279261,0.6337585],
                     [0.4279261,0.6337585,0.1967004],
                     [0.6337585,0.1967004,0.9226179],
                     [0.1967004,0.9226179,-0.7024477],
                     [0.9226179,-0.7024477,0.0131345],
                     [-0.7024477,0.0131345,0.999655],
                     [0.0131345,0.999655,-0.9986200999999999],
                     [0.999655,-0.9986200999999999,-0.9944841999999998],
                     [-0.9986200999999999,-0.9944841999999998,-0.9779978000000001],
                     [-0.9944841999999998,-0.9779978000000001,-0.9129594000000001],
                     [-0.9779978000000001,-0.9129594000000001,-0.6669898],
                     [-0.9129594000000001,-0.6669898,0.1102493],
                     [-0.6669898,0.1102493,0.9756902],
                     [0.1102493,0.9756902,-0.7811937],
                     [0.9756902,-0.7811937,0.7368374],
                     [-0.7811937,0.7368374,-0.7920459],
                     [0.7368374,-0.7920459,0.6854684],
                     [-0.7920459,0.6854684,-0.6613866],
                     [0.6854684,-0.6613866,0.7413643],
                     [-0.6613866,0.7413643,-0.704547]])

    Ytr = np.array([[0.1967004],
                    [0.9226179],
                    [-0.7024477],
                    [0.0131345],
                    [0.999655],
                    [-0.9986200999999999],
                    [-0.9944841999999998],
                    [-0.9779978000000001],
                    [-0.9129594000000001],
                    [-0.6669898],
                    [0.1102493],
                    [0.9756902],
                    [0.7811937],
                    [0.7368374],
                    [-0.7920459],
                    [0.6854684],
                    [-0.6613866],
                    [0.7413643],
                    [-0.704547],
                    [0.3774754]])


    m = 4
    hidden_units = 3

    me = mistura(Xtr, Ytr, m, hidden_units)


    ## Teste

    Ntr = Xtr.shape[0]
    ne = Xtr.shape[1]
    ns = Ytr.shape[1]

    ##add bias
    Xtr = np.concatenate((Xtr, np.ones((Ntr, 1))), axis=1)
    ne = ne + 1

    Wg = me['gating']
    W1 = me['expert_W1']
    W2 = me['expert_W2']
    var = me['expert_var']

    ##calcula saida
    Yg = softmax(np.dot(Xtr, Wg.T))
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

    erro_medio = np.sqrt(np.square(np.sum(Ytr - Ym)) / Ntr)

    print (likelihood)

    errov = Ym - Ytr
    EQMv = 1 / Ntr * np.sum(errov * errov)

    print(EQMv)

    print(Ym)
