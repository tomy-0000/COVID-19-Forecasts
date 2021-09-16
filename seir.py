#%%
# https://oku.edu.mie-u.ac.jp/~okumura/python/sir.html

import matplotlib.pyplot as plt

N = 126000000  # 日本の人口
R0 = 2
gamma = 0.2
beta = R0 * gamma / N

S = N
E = 0 #
I = 1
R = 0

aS = [S]
aE = [E]
aI = [I]
aR = [R]
aRt = [R0]

for t in range(200):
    S, E, I, R = S + m*(N - S) - b*S*I, E + b*S*I - (m + a)*E, I + a*E - (m + g)*I, R + g*I - m*R
    Rt = beta * S / gamma
    aS.append(S)
    aI.append(I)
    aR.append(R)
    aRt.append(Rt)

plt.plot(aS, label="S")
plt.plot(aI, label="I")
plt.plot(aR, label="R")
plt.legend()
