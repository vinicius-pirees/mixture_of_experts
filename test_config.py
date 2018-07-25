import numpy as np
import os
import pandas
from mistura import softmax

filename = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname('treinamento.txt'))) + '/treinamento-1.txt'
series = pandas.read_csv(filename,  header=None)

def series_to_supervised(df, n_lags, n_out=1, dropnan=True):
    """
    Converte uma série temporal para um dataset de aprendizado supervisionado.
    Arguments:
        df: Serie temporal.
        n_lags: Numero de lags (X).
        n_out: Numero de saidas (y).
        dropnan: Remover as linhas superiores com valores Nan.
    Returns:
        Dataframe pandas.
    """
    n_vars = df.shape[1]
    cols, names = list(), list()

    for i in range(n_lags, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

    agg = pandas.concat(cols, axis=1)
    agg.columns = names

    if dropnan:
        agg.dropna(inplace=True)
    return agg



D = series_to_supervised(series, 21).values

X = D[:,0:-1]
Y = D[:,-1].reshape(X.shape[0],1)

train_size = round(X.shape[0] * 0.7) # Dataset de treinamento terá 70% dos dados
test_size = X.shape[0] - train_size
Xv = X[train_size:train_size+test_size,:]
Yv = Y[train_size:train_size+test_size,:]


## Teste

Nv = Xv.shape[0]
ne = Xv.shape[1]
ns = Yv.shape[1]

##add bias
Xv = np.concatenate((Xv, np.ones((Nv, 1))), axis=1)
ne = ne + 1


Wg = np.array([[ 0.67067577,  0.02872399,  0.02634118,  0.39999302,  0.62890172,
         0.40171122,  0.20381844,  0.74692961, -0.14741432,  0.12411937,
         0.76710385,  0.63984003, -0.09741335,  1.00837089,  0.42741529,
         1.11156337,  0.47417913,  0.04635128,  0.10462041,  0.87088133,
         1.54402527,  1.08984899],
       [ 1.2726317 ,  0.79199493,  0.8318971 ,  0.40886801,  0.02236206,
         0.58021876,  0.43555613, -0.65184363,  0.11406486,  0.69667652,
        -0.0591832 , -0.15038106,  0.54714255, -0.53045209,  1.16451188,
         0.17802849,  0.24697656, -0.54411785, -0.24113882,  0.77428588,
         2.18433042, -1.99104664],
       [-0.27924235,  0.43522192,  0.47178526,  0.63333624,  1.53772573,
         0.51580429,  0.33295055,  0.43891381,  0.87396643,  0.3871615 ,
        -0.13649524,  0.42946964, -0.41939443,  0.58890579,  0.08016659,
         0.02454998,  0.20643433,  0.36306822,  0.49329939,  0.45951368,
        -1.21686867,  0.21768272],
       [ 0.28641877,  0.302435  ,  0.5385442 ,  0.22680888,  0.08187221,
         0.32609351,  0.85719653,  0.81093052,  0.55568781,  0.47556765,
         0.41476647,  0.85617456,  0.51108652,  0.739711  ,  0.10358832,
         0.93065832,  1.28184882,  1.4589206 ,  0.85704527,  0.3253324 ,
        -1.23743896,  1.73523056]])


W = {0: np.array([[ 0.09910153,  0.05500486,  0.0441303 , -0.09766039,  0.06286769,
         0.22912749,  0.01857586, -0.06619279, -0.09128526,  0.05289475,
         0.05092102, -0.04069243,  0.03654705,  0.01083512,  0.0816088 ,
         0.00761257, -0.09257461, -0.27113121, -0.14011254,  0.29837283,
        -0.79752026,  0.18315482]]), 1: np.array([[ 0.01860595,  0.0878351 , -0.00477197, -0.01899315, -0.00375673,
        -0.0188798 ,  0.13398292,  0.10408797, -0.1206685 , -0.18434881,
        -0.02464703, -0.0484209 ,  0.03780288,  0.19810835,  0.11909328,
         0.04553871,  0.02402497, -0.01118603,  0.31832529,  0.68279804,
        -0.66517219,  0.04117279]]), 2: np.array([[ 0.02625537, -0.03432095,  0.0014629 ,  0.05343717,  0.01790815,
         0.31408556,  0.40203801,  0.01917167, -0.08454056,  0.05974193,
        -0.00094259, -0.12531308, -0.0559767 ,  0.01858794, -0.10454756,
         0.29536129,  0.25943964,  0.21071884, -0.1868889 ,  0.20058498,
         0.12716301,  0.15730518]]), 3: np.array([[-0.02860845,  0.05795395,  0.00118967, -0.05443803,  0.08289659,
        -0.03329662, -0.1069979 ,  0.03993703,  0.17889596, -0.02348598,
        -0.23594137,  0.07041292, -0.03626119,  0.04654925, -0.04255675,
        -0.08296316, -0.156708  ,  0.16442618, -0.01789787,  0.01746709,
         0.27826207,  0.24803587]])}

var = [0.09753092080469494, 0.05, 0.09453900219945004, 0.2697122435105293]


m = 4

##calcula a saida
Yg = softmax(np.dot(Xv, Wg.T))
Ye = {}
for i in range(m):
    Ye[i] = np.dot(Xv, W[i].T)
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

errov = Ym - Yv
EQMv = 1/Nv * np.sum(errov*errov)


print (likelihood)

print (EQMv)
