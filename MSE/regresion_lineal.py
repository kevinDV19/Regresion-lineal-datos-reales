import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def calcular_error(Theta0, Theta1, X, y):
    n = len(y)
    error_total = 0
    
    for i in range(n):
        error_total += (y[i] - (Theta0 * 1 + Theta1 * X[i])) ** 2
    
    error_promedio = error_total / n
    return error_promedio

def DE_optimizer(X, y):
    dimensions = 2  

    t = np.array([-33871648, 33871648])
    f_range = np.tile(t, (dimensions, 1))

    max_iter = 100

    num_agents = 10
    agents = np.zeros((num_agents, dimensions))

    for i in range(dimensions):
        dim_f_range = f_range[i, 1] - f_range[i, 0]
        agents[:, i] = np.random.rand(num_agents) * dim_f_range + f_range[i, 0]

    best_position = np.zeros(dimensions)
    best_fitness = np.inf
    fitness = np.empty(num_agents)

    for i in range(num_agents):
        theta0, theta1 = agents[i]
        fitness[i] = calcular_error(theta0, theta1, X, y)
        if fitness[i] < best_fitness:
            best_position = agents[i]
            best_fitness = fitness[i]

    initialPop = agents.copy()
    initialFitness = fitness.copy()

    iter = 0

    aux_selector = np.arange(num_agents)

    m = 0.5

    cross_p = 0.2

    # Bucle principal para el proceso de optimización
    while iter < max_iter:
        for i in range(agents.shape[0]):
            # Se eligen tres individuos diferentes
            indexes = aux_selector[aux_selector != i]
            indexes = np.random.choice(indexes, 3, replace=False)
            agents_selected = agents[indexes]

            mut = agents_selected[0] + m * (agents_selected[1] - agents_selected[2])

            prob_vector = np.random.rand(dimensions) <= cross_p
            mut = agents[i] * prob_vector + mut * np.logical_not(prob_vector)

            for j in range(dimensions):
                upper_limit = f_range[j, 1]
                lower_limit = f_range[j, 0]

                if mut[j] < lower_limit:
                    mut[j] = lower_limit
                elif mut[j] > upper_limit:
                    mut[j] = upper_limit

            theta0, theta1 = mut
            fitness_mut = calcular_error(theta0, theta1, X, y)

            if fitness_mut < fitness[i]:
                agents[i] = mut
                fitness[i] = fitness_mut
                if fitness[i] < best_fitness:
                    best_position = agents[i]
                    best_fitness = fitness[i]

            iter = iter + 1
            print("Iteración: " + str(iter))

    
    print("Mejor solución: Theta0 =", best_position[0], ", Theta1 =", best_position[1])
    print("Mejor valor de aptitud:", best_fitness)
    
    xGraph = np.linspace(-33871648, 33871648, 25)
    yGraph = np.linspace(-33871648, 33871648, 25)
    xv, yv = np.meshgrid(xGraph, yGraph)
    fitnessGraph = np.zeros((25, 25))
    for i in range(25):
        for j in range(25):
            arr = [[xv[i, j], yv[i, j]]]
            fitnessGraph[i, j] = calcular_error(arr[0][0], arr[0][1], X, y)
    plt.ion()
    fig = plt.figure(1)
    ax = plt.axes(projection='3d')
    ax.set_xlabel('Theta0')
    ax.set_ylabel('Theta1')
    plt.title('Función de Regresion Lineal', fontsize=20)
    ax.plot_surface(xv, yv, fitnessGraph, alpha=0.6)
    ax.scatter(initialPop[:, 0], initialPop[:, 1], initialFitness[:], c='green', s=10, marker="x")
    ax.scatter(agents[:, 0], agents[:, 1], fitness[:], c='red', s=10, marker="x")
    plt.show(block=True)

df = pd.read_csv('datos.csv', header=None)

x = df.iloc[:, 0].tolist()
y = df.iloc[:, 1].tolist()

print("Arreglo X:", x)
print("Arreglo Y:", y)

DE_optimizer(x, y)