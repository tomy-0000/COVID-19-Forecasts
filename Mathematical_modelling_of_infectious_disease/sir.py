#%%
# https://oku.edu.mie-u.ac.jp/~okumura/python/sir.html

import matplotlib.pyplot as plt

"""
S(t): 感受性
I(t): 感染性
R(t): 回復・隔離

beta: 伝達係数 beta*I(t)は感染力、すなわち感受性人口の単位あたり単位人口あたりの感染率
gamma: 回復率・隔離率

dS(t)/dt = -beta*S(t)*I(t)
    感染した分減る
dI(t)/dt = beta*S(t)*I(t) - gamma*I(t)
    感染した分増える、回復した分減る
dR(t)/dt = gamma*I(t)
    回復した分増える
"""

N = 126000000  # 日本の人口
beta = 0.26 / N
gamma = 0.1
R0 = beta / gamma
delta = 0.25  # 検出率

I = 1 / (delta * gamma)
S = N - I
R = 0

aS = [S]
aI = [I]
aR = [R]
aRt = [R0]


def calc_S(S, I, R):
    return S - beta * S * I


def calc_I(S, I, R):
    return I + beta * S * I - gamma * I


def calc_R(S, I, R):
    return R + gamma * I


for t in range(365):
    S, I, R = calc_S(S, I, R), calc_I(S, I, R), calc_R(S, I, R)
    Rt = beta * S / gamma
    aS.append(S)
    aI.append(I)
    aR.append(R)
    aRt.append(Rt)

plt.plot(aS, label="S")
plt.plot(aI, label="I")
plt.plot(aR, label="R")
plt.legend()
