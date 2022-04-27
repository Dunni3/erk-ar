import numpy as np
import math
import random
from scipy.integrate import odeint

def positive_feedback(x, k, n=1):
    # equal 0 when x = 0
    x_n = np.power(x, n)
    k_n = np.power(k, n)
    return x_n / (k_n + x_n)

def negative_feedback(x, k, n=1):
    # equal 1 when x = 0
    x_n = np.power(x, n)
    k_n = np.power(k, n)
    return k_n / (k_n + x_n)

def system_w_inhibitors(y, t, alpha, beta, k, gamma):
    # y has shape 10
    # alpha and beta have shape 8 - for each of the signaling molecules
    # k has shape 5 for the feedback mechanisms acting on RTK, RAF, ERK, DUSP, SPRY
    # gamma has shape 2 for the two inhibitors
    # 0 -> RTK
    # 1 -> SHP2
    # 2 -> RAS
    # 3 -> RAF
    # 4 -> MEK
    # 5 -> ERK
    # 6 -> DUSP
    # 7 -> SPRY
    # 8 ->  SHP2i
    # 9 ->  MEKi

    dy = np.zeros(y.shape)

    # compute MEKi and SHP2i inhibitor complexation rates
    shp2i_compelxation_rate = gamma[1] * y[1] * y[8]
    meki_complexation_rate = gamma[0] * y[4] * y[9]

    # model RTK concentration
    dy[0] = alpha[0] * negative_feedback(y[7], k[0], n=3) - beta[0] * y[0]

    # model SHP2
    dy[1] = alpha[1] * y[0] - beta[1] * y[1] - shp2i_compelxation_rate

    # model RAS
    dy[2] = alpha[2] * y[1] - beta[2] * y[2]

    # model RAF
    dy[3] = alpha[3] * y[2] * negative_feedback(y[5], k[1], n=3) - beta[3] * y[3]

    # model MEK
    dy[4] = alpha[4] * y[3] - beta[4] * y[4] - meki_complexation_rate

    # model ERK
    dy[5] = alpha[5] * negative_feedback(y[6], k[2], n=3) * y[4] - beta[5] * y[5]

    # model DUSP
    dy[6] = alpha[6] * positive_feedback(y[5], k[3], n=3) - beta[6] * y[6]

    # model SPRY
    dy[7] = alpha[7] * positive_feedback(y[5], k[4], n=3) - beta[7] * y[7]

    # model SHP2i
    dy[8] = -shp2i_compelxation_rate

    # model MEKi
    dy[9] = -meki_complexation_rate

    return dy

def move(args):
    # randomly change the system parameters
    alpha, beta, k, gamma = args

    parameters = list()
    parameters.append(alpha)
    parameters.append(beta)
    parameters.append(k)
    parameters.append(gamma)

    p_idx = random.randint(0, len(parameters)-1)
    p = parameters[p_idx]

    s_idx = random.randint(0, len(p)-1)

    a = random.randint(1, 10)
    p[s_idx] = a

    alpha, beta, k, gamma = parameters
    return (alpha, beta, k, gamma,)


def simulated_annealing(model, t, y0, args, Tmax, Tmin, iter, maxsteps):

    T_cur = Tmax
    T_min = Tmin
    delta_T = 0

    while T_cur >=T_min:
        for i in range(iter):
            e = evaluate_fitness(model, t, y0, args)

            new_args = move(args)
            eNew = evaluate_fitness(model, t, y0, new_args)

            print('T:{} error: {}'.format(T_cur, e))
            if eNew - e < 0:
                args = new_args
            else:
                #metropolis principle
                p = math.exp(-10 * (eNew-e) / T_cur)
                r = np.random.uniform(low=0, high=1)
                if r < p:
                    args = new_args

        delta_T += 0.05
        T_cur = T_cur - delta_T

    print('optimal',args)

def cost(y1, y2):
    # In Supplementary Table 2
    # Fit the data with pMEK reduction

    pMEK_1 = y1[:,4][-1]
    pMEK_2 = y2[:,4][-1]

    # e.g use the first row data - MIA PaCa-2
    r = (pMEK_1 - pMEK_2) / pMEK_1

    error = abs(r - 0.7)

    return error

def evaluate_fitness(model, t, y0, args):
    # TODO: Base system
    y_base = odeint(model, t=t, y0=y0, args=args)

    # TODO: system with MEK inhibitor
    alpha, beta, k, gamma = args
    gamma = gamma * 0.5

    args = (alpha, beta, k, gamma,)

    # getting steady-state values of base system
    base_ss_levels = y_base[-1, :]

    # set initial conditions
    y0 = base_ss_levels
    y0[9] = 500

    y1 = odeint(model, t=t, y0=y0, args=args)

    # TODO: system with MEK and SHP2 inhibitor
    y0[9] = 500
    y0[8] = 100

    y2 = odeint(model, t=t, y0=y0, args=args)

    error = cost(y1, y2)

    return error

def main():
    # set time points
    t_base = np.linspace(0, 30, 300)

    # set initial conditions
    y0 = np.zeros(10)

    # set parameters
    alpha = np.ones(y0.shape[0] - 2) * 2
    beta = np.ones(y0.shape[0] - 2)
    k = np.ones(5) * 2
    # k[0] = 3 # RTK regulation
    # k[3] = 10 # how high can [ERK] get before DUSP is produced..which modulates ERK production
    gamma = np.ones(2)
    sys_args = (alpha, beta, k, gamma,)

    simulated_annealing(system_w_inhibitors, t=t_base, y0=y0, args=sys_args,
        Tmax=200.0, Tmin=0.95, iter=1, maxsteps=200)


    return

if __name__ == '__main__':
    main()