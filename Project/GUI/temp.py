import matplotlib.pyplot as plt
import math

def get_alpha(x):
    raw_alpha = x

    if raw_alpha <= 50:
        alpha = 1.0 * (raw_alpha - 0) / (50 - 0) # [0, 1]
        alpha = alpha * alpha # [0, 1] exponential
    else:
        alpha = 1.0 * (raw_alpha - 50) / (100 - 50) # [0, 1]
        alpha *= 15 # [0, 15] exponential
        alpha = alpha * alpha # [0, 225] exponential
        alpha += 1 # [1, 225] exponential

    return alpha

def get_alpha2(x):
    raw_alpha = x

    if raw_alpha <= 50:
        alpha = 1.0 * (raw_alpha - 0) / (50 - 0) # [0, 1]
        alpha = math.sqrt(alpha) # [0, 1] exponential
    else:
        alpha = 1.0 * (raw_alpha - 50) / (100 - 50) # [0, 1]
        alpha *= 6 # [0, 15] exponential
        alpha = math.exp(alpha) # [0, 225] exponential
        alpha += 1 # [1, 225] exponential

    return alpha


def get_alpha3(x):
    raw_alpha = x

    if raw_alpha <= 50:
        alpha = 1.0 * (raw_alpha - 0) / (50 - 0) # [0, 1]
        alpha = math.log(alpha+1e-2) # [0, 1] exponential
    else:
        alpha = 1.0 * (raw_alpha - 50) / (100 - 50) # [0, 1]
        alpha *= 6 # [0, 15] exponential
        alpha = math.exp(alpha) # [0, 225] exponential
        alpha += 1 # [1, 225] exponential

    return alpha


def get_beta(x):
    raw_beta = x

    beta = 1.0 * (raw_beta - 0) / (100 - 0) # [0, 1]
    beta *= 300 # [0, 300]
    beta -= 150 # [-150, 150]

    return beta


# x = range(50, 100)
x = range(0, 50)

# plt.plot(x, x)

x2 = [get_alpha(a) for a in x]
plt.plot(x, x2, color='red',  label='alpha')

x2 = [get_alpha2(a) for a in x]
plt.plot(x, x2, color='blue',  label='alpha2')

x2 = [get_alpha3(a) for a in x]
plt.plot(x, x2, color='green',  label='alpha3')

# x2 = [get_beta(a) for a in x]
# plt.plot(x, x2, color='green', label='beta,bright')
plt.legend()
plt.show()