import numpy as np
import time


#Golf Optimization Algorithm (GOA)

def GOA(X, fitness, lowerbound, upperbound, Max_iterations):

    [SearchAgents, dimension] = X.shape

    lowerbound = np.ones(dimension) * lowerbound
    upperbound = np.ones(dimension) * upperbound

    # Initialize population
    # X = np.random.uniform(lowerbound, upperbound, size=(SearchAgents, dimension))

    # Evaluate initial population
    fit = np.zeros(SearchAgents)
    for i in range(SearchAgents):
        L = X[i, :]
        fit[i] = fitness(L)

    # Initialize best solution
    Xbest = X[fit.argmin()]
    fbest = fit.min()
    ct = time.time()
    GOA_curve = np.zeros(Max_iterations)  # Store best fitness over iterations

    for t in range(Max_iterations):
        for i in range(SearchAgents):
            # Phase 1: Exploration (global search)
            if np.random.rand() < 0.5:
                I = np.round(1 + np.random.rand())
                RAND = np.random.rand()
            else:
                I = np.round(1 + np.random.rand(dimension))
                RAND = np.random.rand(dimension)

            X_P1 = X[i, :] + RAND * (Xbest - I * X[i, :])  # Eq. (4)
            X_P1 = np.maximum(X_P1, lowerbound)
            X_P1 = np.minimum(X_P1, upperbound)
            L = X_P1
            F_P1 = fitness(L)
            if F_P1 < fit[i]:
                X[i, :] = X_P1
                fit[i] = F_P1

            # Phase 2: Exploitation (local search)
            X_P2 = X[i, :] + (1 - 2 * np.random.rand()) * (
                        lowerbound / t + np.random.rand() * (upperbound / t - lowerbound / t))  # Eq. (6)
            X_P2 = np.maximum(X_P2, lowerbound / t)
            X_P2 = np.minimum(X_P2, upperbound / t)
            X_P2 = np.maximum(X_P2, lowerbound)
            X_P2 = np.minimum(X_P2, upperbound)
            L = X_P2
            F_P2 = fitness(L)
            if F_P2 < fit[i]:
                X[i, :] = X_P2
                fit[i] = F_P2

        # Update best solution
        if fit.min() < fbest:
            fbest = fit.min()
            Xbest = X[fit.argmin()]

        GOA_curve[t] = fbest
    ct = time.time() - ct
    return fbest, GOA_curve, Xbest, ct

