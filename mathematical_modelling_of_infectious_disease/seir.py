#%%
import matplotlib.pyplot as plt

"""
S(t): 感受性
E(t): 潜伏期
I(t): 感染性
R(t): 回復・隔離

beta: 伝達係数 beta*I(t)は感染力、すなわち感受性人口の単位あたり単位人口あたりの感染率
epsilon: 潜伏期Eから感染性Iへの遷移率
gamma: 回復率・除去率

dS(t)/dt = -beta*S(t)*I(t)
    感染した分減る
dE(t)/dt = beta*S(t)*I(t) - epsilon*E(t)
    感染した分増える、発症した分減る
dI(t)/dt = epsilon*E(t) - gamma*I(t)
    発症した分増える、回復した分減る
dR(t)/dt = gamma*I(t)
    回復した分増える
"""

N = 126000000  # 日本の人口
beta = 0.26 / N
epsilon = 0.2
gamma = 0.1
R0 = beta / gamma
delta = 0.25  # 検出率

I = 1 / (delta * gamma)
S = N - I
E = 0
R = 0

aS = [S]
aE = [E]
aI = [I]
aR = [R]
aRt = [R0]


def calc_S(S, E, I, R):
    return S - beta * S * I


def calc_E(S, E, I, R):
    return E + beta * S * I - epsilon * E


def calc_I(S, E, I, R):
    return I + epsilon * E - gamma * I


def calc_R(S, E, I, R):
    return R + gamma * I


for t in range(365):
    S, E, I, R = (
        calc_S(S, E, I, R),
        calc_E(S, E, I, R),
        calc_I(S, E, I, R),
        calc_R(S, E, I, R),
    )
    Rt = beta * S / gamma
    aS.append(S)
    aE.append(E)
    aI.append(I)
    aR.append(R)
    aRt.append(Rt)

plt.plot(aS, label="S")
plt.plot(aE, label="E")
plt.plot(aI, label="I")
plt.plot(aR, label="R")
plt.legend()
