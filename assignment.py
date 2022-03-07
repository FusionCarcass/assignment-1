from enum import Enum
import argparse
import time

# Third party libraries
import mlrose_hiive as mlrose
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Import datasets
from sklearn import datasets
from sklearn.datasets import load_iris

class Experiments(Enum):
    queens = 'queens'
    sixpeaks = 'sixpeaks'
    discrete = 'discrete'
    neural = 'neural'
    iris = 'iris'
    knapsack = 'knapsack'
    knapsackstart = 'knapsackstart'
    flipflop = 'flipflop'
    
    def __str__(self):
        return self.value

parser = argparse.ArgumentParser(description='Trains a model using the ember dataset.')
parser.add_argument('--experiment', type=Experiments, default=Experiments.discrete, choices=list(Experiments))

# Define alternative N-Queens fitness function for maximization problem
def queens_max(state):
    # Initialize counter
    fitness_cnt = 0

    # For all pairs of queens
    for i in range(len(state) - 1):
        for j in range(i + 1, len(state)):

            # Check for horizontal, diagonal-up and diagonal-down attacks
            if state[j] != state[i] \
                and (state[j] != state[i] + (j - i)) \
                and (state[j] != state[i] - (j - i)):

                # If no attacks, then increment counter
                fitness_cnt += 1

    return fitness_cnt

#This is the example code from the documentation (https://mlrose.readthedocs.io/en/stable/source/tutorial1.html#solving-optimization-problems-with-mlrose)
def queens():
    fitness = mlrose.Queens()
    
    # Initialize custom fitness function object
    fitness_cust = mlrose.CustomFitness(queens_max)
    
    problem = mlrose.DiscreteOpt(length = 8, fitness_fn = fitness, maximize = False, max_val = 8)
    
    # Define decay schedule
    schedule = mlrose.ExpDecay()

    # Define initial state
    init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])

    # Solve problem using simulated annealing
    best_state, best_fitness, k = mlrose.simulated_annealing(problem, schedule = schedule,
                                                          max_attempts = 10, max_iters = 1000,
                                                          init_state = init_state, random_state = 1,
                                                          curve = True)

    print(best_state)
    print(best_fitness)
    
    print(k)
    plt.plot(k[:,1], k[:,0])
    plt.show()

def sixpeaks():
    fitness = mlrose.SixPeaks(t_pct=0.25)
    
    problem = mlrose.DiscreteOpt(length = 12, fitness_fn = fitness, maximize = True, max_val = 2)

    # Solve problem using hill climb
    best_state, best_fitness, k = mlrose.hill_climb(problem, max_iters = 500,
                                                  random_state = 42, curve = True)

    print(best_state)
    print(best_fitness)
    
    print(k)
    plt.plot(k[:,1], k[:,0])
    plt.show()
    
def discrete():
    #Additional Parameters for the simulated annealing algorithm
    schedule = mlrose.ExpDecay()
    
    #Additional parameters for the knapsack problem
    weights = [10, 1, 10, 13, 18, 5, 13, 5, 16, 8, 17]
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    
    #Additional parameters for the TSP
    coords = [(0, 0), (3, 0), (3, 2), (2, 4), (1, 3)]
    dists = [(0, 1, 3), (0, 2, 5), (0, 3, 1), (0, 4, 7), (1, 3, 6),
             (4, 1, 9), (2, 3, 8), (2, 4, 2), (3, 2, 8), (3, 4, 4)]
    
    problems = [
        mlrose.DiscreteOpt(length = 50, fitness_fn = mlrose.SixPeaks(t_pct=0.2), maximize = True, max_val = 2),
        #mlrose.DiscreteOpt(length = len(weights), fitness_fn = mlrose.Knapsack(weights, values, 0.6), maximize = True, max_val = len(weights)),
        #mlrose.DiscreteOpt(length = 10, fitness_fn = mlrose.FlipFlop(), maximize = True, max_val = 2)
    ]
    
    names = [
        "Six Peaks",
        #"Knapsack",
        #"Flipflop"
    ]
    
    #Initial state
    example = [0, 1, 0, 0, 1, 1, 0, 0, 0, 0]
    
    for problem,name in zip(problems, names):
        # Solve problem using hill climb
        #NOTE: Using an initial state with random restart doesn't change anything because it restarts to the same state
        best_state, best_fitness, best_k = None, -100, None
        for i in range(0, 1):
            t0 = time.time()
            state, fitness, k = mlrose.random_hill_climb(problem, max_attempts = 300, max_iters = 5000, restarts = 1500, curve = True)
            t1 = time.time()
            print(f"Iteration:{i} Time:{t1-t0} Score:{fitness}")
            if fitness > best_fitness:
                best_state = state
                best_fitness = fitness
                best_k = k
        
        best_k = (np.arange(1, best_k.shape[0] + 1), best_k[:,0], "Random Hill Climb")
        results = [best_k]
        print("#########################")
        print("#Random Hill Climbing")
        print("#########################")
        print(best_state)
        print(best_fitness)
        print(best_k)
        print()
        
        best_state, best_fitness, best_k = None, -100, None
        for i in range(0, 200):
            t0 = time.time()
            state, fitness, k = mlrose.simulated_annealing(problem, schedule = schedule, max_attempts = 300, max_iters = 5000, curve = True)
            t1 = time.time()
            print(f"Iteration:{i} Time:{t1-t0} Score:{fitness}")
            if fitness > best_fitness:
                best_state = state
                best_fitness = fitness
                best_k = k
                
        best_k = (np.arange(1, best_k.shape[0] + 1), best_k[:,0], "Simulated Annealing")
        results.append(best_k)
        print("#########################")
        print("#Simulated Annealing")
        print("#########################")
        print(best_state)
        print(best_fitness)
        print(best_k)
        print()
        
        best_state, best_fitness, k = mlrose.genetic_alg(problem, pop_size=500, mutation_prob=0.05, max_attempts=20, max_iters=1000, curve=True, random_state=None)
        k = (np.arange(1, k.shape[0] + 1), k[:,0], "Genetic Algorithm")
        results.append(k)
        print("#########################")
        print("#Genetic Algorithm")
        print("#########################")
        print(best_state)
        print(best_fitness)
        print(k)
        print()
        
        best_state, best_fitness, k = mlrose.mimic(problem, pop_size=200, keep_pct=0.2, max_attempts=10, max_iters=1000, curve=True, random_state=None)
        k = (np.arange(1, k.shape[0] + 1), k[:,0], "MIMIC")
        results.append(k)
        print("#########################")
        print("#MIMIC")
        print("#########################")
        print(best_state)
        print(best_fitness)
        print(k)
        print()
        
        fig = plt.figure()
        for result in results:
            plt.plot(result[0], result[1], label=result[2])
        plt.xlabel("Iterations")
        plt.ylabel("Score")
        
        
        # Shrink current axis's height by 10% on the bottom
        #box = ax.get_position()
        #ax.set_position([box.x0, box.y0 + box.height * 0.1,
        #                 box.width, box.height * 0.9])

        # Put a legend below current axis
        #ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.05),
        #          fancybox=True, shadow=True, ncol=5)
        
        labels = ["Random Hill Climbing", "Simulated Annealing", "Genetic Algorithm", "MIMIC"]
        plt.legend(labels, loc="center", bbox_to_anchor=(0.5, -0.3), ncol=2)
        fig.subplots_adjust(bottom=0.25)
        plt.title(f"Comparison of Algorithms on {name}")
        plt.show()
        
        #plt.savefig('example.png', 
        #    dpi=300, 
        #    format='png', 
        #    bbox_extra_artists=(lg,), 
        #    bbox_inches='tight')

def gen_valid_state(choices, probabilities, fitness):
    item = np.random.choice(choices, probabilities.shape[0], p=probabilities)
    while fitness.evaluate(item) <= 0:
        item = np.random.choice(choices, probabilities.shape[0], p=probabilities)
    
    return item

def knapsack():
    #Additional Parameters for the simulated annealing algorithm
    schedule = mlrose.ExpDecay()
    
    #Additional parameters for the knapsack problem
    weights = [1, 1, 8, 1, 9, 4, 2, 6, 6, 2, 4]
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    
    npweights = np.array(weights)
    total = np.sum(npweights)
    probabilities = npweights/total
    inverse = 1.0 / probabilities
    probabilities = inverse / np.sum(inverse)
    choices = np.arange(0, npweights.shape[0])
    maximum = 0.6 * total
    
    fitness_fn = mlrose.Knapsack(weights, values, 2.5)
    problem = mlrose.DiscreteOpt(length = len(weights), fitness_fn = fitness_fn, maximize = True, max_val = len(weights))
    
    name = "Knapsack"
    
    # Solve problem using hill climb
    #NOTE: Using an initial state with random restart doesn't change anything because it restarts to the same state
    best_state, best_fitness, best_k = None, -100, None
    for i in range(0, 1):
        t0 = time.time()
        state, fitness, k = mlrose.random_hill_climb(problem, init_state=gen_valid_state(choices, probabilities, fitness_fn), max_attempts = 250, max_iters = 1500, restarts = 1500, curve = True)
        t1 = time.time()
        print(f"Iteration:{i} Time:{t1-t0} Score:{fitness}")
        if fitness > best_fitness:
            best_state = state
            best_fitness = fitness
            best_k = k
    
    best_k = (np.arange(1, best_k.shape[0] + 1), best_k[:,0], "Random Hill Climb")
    results = [best_k]
    print("#########################")
    print("#Random Hill Climbing")
    print("#########################")
    print(best_state)
    print(best_fitness)
    print(best_k)
    print()
    
    best_state, best_fitness, best_k = None, -100, None
    for i in range(0, 100):
        t0 = time.time()
        state, fitness, k = mlrose.simulated_annealing(problem, init_state=gen_valid_state(choices, probabilities, fitness_fn), schedule = schedule, max_attempts = 200, max_iters = 1500, curve = True)
        t1 = time.time()
        print(f"Iteration:{i} Time:{t1-t0} Score:{fitness}")
        if fitness > best_fitness:
            best_state = state
            best_fitness = fitness
            best_k = k
            
    best_k = (np.arange(1, best_k.shape[0] + 1), best_k[:,0], "Simulated Annealing")
    results.append(best_k)
    print("#########################")
    print("#Simulated Annealing")
    print("#########################")
    print(best_state)
    print(best_fitness)
    print(best_k)
    print()
    
    best_state, best_fitness, k = mlrose.genetic_alg(problem, pop_size=500, mutation_prob=0.05, max_attempts=20, max_iters=1000, curve=True, random_state=None)
    k = (np.arange(1, k.shape[0] + 1), k[:,0], "Genetic Algorithm")
    results.append(k)
    print("#########################")
    print("#Genetic Algorithm")
    print("#########################")
    print(best_state)
    print(best_fitness)
    print(k)
    print()
    
    best_state, best_fitness, k = mlrose.mimic(problem, pop_size=200, keep_pct=0.2, max_attempts=10, max_iters=1000, curve=True, random_state=None)
    k = (np.arange(1, k.shape[0] + 1), k[:,0], "MIMIC")
    results.append(k)
    print("#########################")
    print("#MIMIC")
    print("#########################")
    print(best_state)
    print(best_fitness)
    print(k)
    print()
    
    fig = plt.figure()
    for result in results:
        plt.plot(result[0], result[1], label=result[2])
    plt.xlabel("Iterations")
    plt.ylabel("Score")
    
    
    # Shrink current axis's height by 10% on the bottom
    #box = ax.get_position()
    #ax.set_position([box.x0, box.y0 + box.height * 0.1,
    #                 box.width, box.height * 0.9])

    # Put a legend below current axis
    #ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.05),
    #          fancybox=True, shadow=True, ncol=5)
    
    labels = ["Random Hill Climbing", "Simulated Annealing", "Genetic Algorithm", "MIMIC"]
    plt.legend(labels, loc="center", bbox_to_anchor=(0.5, -0.3), ncol=2)
    fig.subplots_adjust(bottom=0.25)
    plt.title(f"Comparison of Algorithms on {name}")
    plt.show()
    
    #plt.savefig('example.png', 
    #    dpi=300, 
    #    format='png', 
    #    bbox_extra_artists=(lg,), 
    #    bbox_inches='tight')

def flipflop():
    #Additional Parameters for the simulated annealing algorithm
    schedule = mlrose.ExpDecay()
    
    fitness_fn = mlrose.FlipFlop()
    problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness_fn, maximize = True, max_val = 2)
    
    name = "Flipflop"
    
    # Solve problem using hill climb
    #NOTE: Using an initial state with random restart doesn't change anything because it restarts to the same state
    best_state, best_fitness, best_k = None, -100, None
    for i in range(0, 1):
        t0 = time.time()
        state, fitness, k = mlrose.random_hill_climb(problem, max_attempts = 250, max_iters = 1500, restarts = 1500, curve = True)
        t1 = time.time()
        print(f"Iteration:{i} Time:{t1-t0} Score:{fitness}")
        if fitness > best_fitness:
            best_state = state
            best_fitness = fitness
            best_k = k
    
    best_k = (np.arange(1, best_k.shape[0] + 1), best_k[:,0], "Random Hill Climb")
    results = [best_k]
    print("#########################")
    print("#Random Hill Climbing")
    print("#########################")
    print(best_state)
    print(best_fitness)
    print(best_k)
    print()
    
    best_state, best_fitness, best_k = None, -100, None
    for i in range(0, 100):
        t0 = time.time()
        state, fitness, k = mlrose.simulated_annealing(problem, schedule = schedule, max_attempts = 200, max_iters = 1500, curve = True)
        t1 = time.time()
        print(f"Iteration:{i} Time:{t1-t0} Score:{fitness}")
        if fitness > best_fitness:
            best_state = state
            best_fitness = fitness
            best_k = k
            
    best_k = (np.arange(1, best_k.shape[0] + 1), best_k[:,0], "Simulated Annealing")
    results.append(best_k)
    print("#########################")
    print("#Simulated Annealing")
    print("#########################")
    print(best_state)
    print(best_fitness)
    print(best_k)
    print()
    
    best_state, best_fitness, k = mlrose.genetic_alg(problem, pop_size=500, mutation_prob=0.05, max_attempts=20, max_iters=1000, curve=True, random_state=None)
    k = (np.arange(1, k.shape[0] + 1), k[:,0], "Genetic Algorithm")
    results.append(k)
    print("#########################")
    print("#Genetic Algorithm")
    print("#########################")
    print(best_state)
    print(best_fitness)
    print(k)
    print()
    
    best_state, best_fitness, k = mlrose.mimic(problem, pop_size=800, keep_pct=0.2, max_attempts=10, max_iters=1000, curve=True, random_state=None)
    k = (np.arange(1, k.shape[0] + 1), k[:,0], "MIMIC")
    results.append(k)
    print("#########################")
    print("#MIMIC")
    print("#########################")
    print(best_state)
    print(best_fitness)
    print(k)
    print()
    
    fig = plt.figure()
    for result in results:
        plt.plot(result[0], result[1], label=result[2])
    plt.xlabel("Iterations")
    plt.ylabel("Score")
    
    
    # Shrink current axis's height by 10% on the bottom
    #box = ax.get_position()
    #ax.set_position([box.x0, box.y0 + box.height * 0.1,
    #                 box.width, box.height * 0.9])

    # Put a legend below current axis
    #ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.05),
    #          fancybox=True, shadow=True, ncol=5)
    
    labels = ["Random Hill Climbing", "Simulated Annealing", "Genetic Algorithm", "MIMIC"]
    plt.legend(labels, loc="center", bbox_to_anchor=(0.5, -0.3), ncol=2)
    fig.subplots_adjust(bottom=0.25)
    plt.title(f"Comparison of Algorithms on {name}")
    plt.show()
    
    #plt.savefig('example.png', 
    #    dpi=300, 
    #    format='png', 
    #    bbox_extra_artists=(lg,), 
    #    bbox_inches='tight')

def neural():
    # Load the dataset
    covertypes = datasets.fetch_covtype()
    
    #Fix issue with labels (go from 1 - 7 to 0 - 6)
    covertypes.target = covertypes.target - 1
    
    # Convert labels to one-hot encoded arrays
    print("Type:", type(covertypes.target))
    temp = np.zeros((covertypes.target.size, covertypes.target.max() + 1))
    temp[np.arange(covertypes.target.size), covertypes.target] = 1
    covertypes.target = temp
    print("Temp.Shape:", temp.shape)
    print("Temp:", temp)
    
    # Split the data into train, validation, and test sets
    x_train, x_test, y_train, y_test = train_test_split(covertypes.data, covertypes.target, test_size=0.1,random_state=0)
    
    # Initialize neural network object and fit object
    #random_hill_climb, gradient_descent
    nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [4], activation = 'relu', \
                                     algorithm = 'random_hill_climb', max_iters = 1000, \
                                     bias = True, is_classifier = True, learning_rate = 0.0001, \
                                     early_stopping = True, clip_max = 5, max_attempts = 100, \
                                     random_state = 3, curve=True)

    t0 = time.time()
    
    nn_model1.fit(x_train, y_train)
    
    t1 = time.time()
    
    print("Total Time:", t1 - t0)
    
    # Predict labels for train set and assess accuracy
    train_pred = nn_model1.predict(x_train)
    train_accuracy = accuracy_score(y_train, train_pred)

    # Predict labels for test set and assess accuracy
    test_pred = nn_model1.predict(x_test)
    test_accuracy = accuracy_score(y_test, test_pred)

    print("Train Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)
    print(nn_model1.fitness_curve)
    
    results = []
    results.append((np.arange(0, nn_model1.fitness_curve.shape[0]), nn_model1.fitness_curve, "Gradient Descent"))
    
    fig = plt.figure()
    for result in results:
        plt.plot(result[0], result[1], label=result[2])
    plt.xlabel("Iterations")
    plt.ylabel("Score")
    
    labels = ["Random Hill Climbing", "Simulated Annealing", "Genetic Algorithm", "MIMIC"]
    plt.legend(labels, loc="center", bbox_to_anchor=(0.5, -0.3), ncol=2)
    fig.subplots_adjust(bottom=0.25)
    plt.title(f"Comparison of Algorithms on Neural Network Weight Optimization")
    plt.show()

#This code started from https://github.com/hiive/mlrose/blob/master/tutorial_examples.ipynb to learn how to use the library
#I heavily modified it in order to generate the data I needed.
def iris():
    data = load_iris()
    
    print(data.data[0])
    print(np.min(data.data, axis = 0))
    print(np.max(data.data, axis = 0))
    print(np.unique(data.target))
    
    one_hot = OneHotEncoder()
    data.target = one_hot.fit_transform(data.target.reshape(-1, 1)).todense()
    scaler = MinMaxScaler()
    data.data = scaler.fit_transform(data.data)
    
    print("Input Shape:", data.data.shape)
    print("Output Shape:", data.target.shape)
    
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size = 0.2, random_state = 3)
    print("x_train.shape:", x_train.shape)
    print("y_train.shape:", y_train.shape)
    print("x_test.shape:", x_test.shape)
    print("y_test.shape:", y_test.shape)

    results = []
    for algorithm in ['gradient_descent', 'random_hill_climb', 'simulated_annealing', 'genetic_alg']:
        # Initialize neural network object and fit object - attempt 1
        learning_rate = 0.0001
        if algorithm == 'random_hill_climb' or algorithm == 'simulated_annealing':
            learning_rate = 0.1
            
        model = mlrose.NeuralNetwork(hidden_nodes = [24], activation ='sigmoid',
                                         algorithm = algorithm,
                                         max_iters = 2000, bias = True, is_classifier = True, restarts=75,
                                         learning_rate = learning_rate, early_stopping = True, pop_size = 800,
                                         clip_max = 2, max_attempts = 125, curve=True)

        print("Algorithm:", algorithm)
        t0 = time.time()
        model.fit(x_train, y_train)
        t1 = time.time()
        print("Time:", t1 - t0)
        
        # Predict labels for train set and assess accuracy
        train_predictions = model.predict(x_train)
        train_accuracy = accuracy_score(y_train, train_predictions)
        test_predictions = model.predict(x_test)
        test_accuracy = accuracy_score(y_test, test_predictions)
        print(f"Algorithm:{algorithm} Train:{train_accuracy} Test:{test_accuracy}")
        print("Loss:", model.fitness_curve)
        print("Weights:")
        print(model.fitted_weights)
        if algorithm == 'gradient_descent':
            results.append((algorithm, model.fitness_curve))
        else:
            results.append((algorithm, model.fitness_curve[:,0]))
    
    fig = plt.figure()
    for result in results:
        if result[0] != 'gradient_descent':
            plt.plot(-result[1], label=result[0])
        else:
            plt.plot(result[1], label=result[0])
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    
    labels = ["Gradient Descent", "Random Hill Climbing"]
    plt.legend(labels, loc="center", bbox_to_anchor=(0.5, -0.3), ncol=2)
    fig.subplots_adjust(bottom=0.25)
    plt.title(f"Comparison of Algorithms on Neural Network Weight Optimization")
    plt.show()

def knapsackstart():
    state = np.array([6, 4, 9, 6, 9, 9, 4, 9, 2, 2, 6])
    #state = np.array([1, 1, 8, 1, 9, 4, 2, 6, 6, 2, 4])
    capacity = np.sum(state) * 0.6
    
    total = np.sum(state)
    probabilities = state/total
    inverse = 1.0 / probabilities
    probabilities = inverse / np.sum(inverse)
    choices = np.arange(0, state.shape[0])
    
    iterations = 100000
    valid = 0
    for i in range(0, iterations):
        #temp = np.random.randint(0, high=state.shape[0], size=state.shape[0])
        temp = np.random.choice(choices, state.shape[0], p=probabilities)
        total = np.sum(state[temp])
        #print("Total:", total, "Capacity:", capacity, "Permutation:", temp)
        if total < capacity:
            valid += 1
            
    percentage = valid / iterations
    print(f"The percent of valid starting states is approximately {percentage} based on a capacity of {capacity}.")

if __name__ == '__main__':
    # Parse the arguments
    global args
    args = parser.parse_args()
    
    if args.experiment == Experiments.queens:
        queens()
    elif args.experiment == Experiments.sixpeaks:
        sixpeaks()
    elif args.experiment == Experiments.discrete:
        discrete()
    elif args.experiment == Experiments.neural:
        neural()
    elif args.experiment == Experiments.iris:
        iris()
    elif args.experiment == Experiments.knapsack:
        knapsack()
    elif args.experiment == Experiments.knapsackstart:
        knapsackstart()
    elif args.experiment == Experiments.flipflop:
        flipflop()
    else:
        raise Exception(f"[!] The experiment {args.experiment} has not been implemented yet.")