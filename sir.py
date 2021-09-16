#%%
import matplotlib.pyplot as plt

N = 126000000  # 日本の人口
R0 = 2
gamma = 0.2
beta = R0 * gamma / N

S = N
I = 1  # 最初は1人が感染
R = 0

aS = [S]
aI = [I]
aR = [R]
aRt = [R0]

for t in range(200):
    S, I, R = S - beta * S * I, I + beta * S * I - gamma * I, R + gamma * I
    Rt = beta * S / gamma
    aS.append(S)
    aI.append(I)
    aR.append(R)
    aRt.append(Rt)

plt.plot(aS, label="S")
plt.plot(aI, label="I")
plt.plot(aR, label="R")
plt.legend()
