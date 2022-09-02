import numpy as np
import sys
from numpy import random
from nonogram import Game, Rules, checkSolution
from util import readRulesFile, printSol, createConstraints, fitness as evaluateFitness
import matplotlib.pyplot as plt
import time

class Solution:
    def __init__(self, points, constraints):
        self.points = points
        self.fitness = evaluateFitness(points, constraints)

def main(nPopulation=500 , puzzleName='input' , prob=(0.4/100), portionP=0.2 , portionC=0.5):
    rules = readRulesFile('puzzles/' + puzzleName + '.txt')  #rules contain row and col constraints
    constraints = createConstraints(rules, nPopulation)      #Count number of to be colored homes
    rules, nLines, nColumns, nPoints, nPopulation = constraints
    constraints = rules, nLines, nColumns, nLines * nColumns, nPopulation



    #For Second Fitness Function
    # a = list()
    # combination=list()
    # result = list()
    # for i in range(1):
    #     print(rules.lines[i])
    #     result[i]=(make_combination(rules.lines[i],nColumns,0,i,0,a,combination))
    # print(combination)


    mySol , itr = GA(constraints , prob, portionP , portionC)
    print(checkSolution(Game(nLines, nColumns, mySol.points), rules))
    printSol(mySol, constraints)
    return itr
global iterations

def make_combination(rules,nColumns , index , row , constraint_number , selected , combination):
    # print(rules.lines)
    #       print(index)
          print(selected)
          summ=0
          if(len(rules)==constraint_number):
              print("inside if")
              print(selected)
              combination.add(selected)
              return
          if (index < nColumns):
              summ = sum(rules) + len(rules)-1
              if(index+summ < nColumns):
                t1 = list()
                t2 = list()
                # print(selected)
                for e in selected:

                    t1.append(e)
                    t2.append(e)
                # print(t2)
                # t1 = selected
                # t2 = selected
                make_combination(rules , nColumns , index+1 , row , constraint_number , t1 , combination )
                t2.append(index)
                print(t2)
                make_combination(rules , nColumns , index+rules[constraint_number]+1 , row , constraint_number+1 , t2 , combination)
                selected.clear()
                # selected.clear()
              # return selected


def GA(constraints , prob=(0.4/100),  portionP=0.2 , portionC=0.5):
    rules, nLines, nColumns, nPoints, nPopulation = constraints
    iterations=0
    P = randomSolutions(constraints)       #construct population at once
    while not converge(P, constraints):    #Terminate state
        PP = crossover(P, constraints)     #Build(500) children
        PPP = mutation(PP, constraints , prob)
        P = select(P, PPP, constraints , portionP , portionC)
        iterations += 1
        print(iterations)
        print(P[0].fitness)
        printSol(P[0], constraints)
        if iterations>500:
            break

    return (best(P, constraints) , iterations)
def randomSolutions(constraints):                           #construct initial population
    rules, nLines, nColumns, nPoints, nPopulation = constraints
    S = []
    # print()
    for _ in range(nPopulation):                        #each population contain colored map
        s = []
        for _ in range(nPoints):
            if random.random() <= 0.5:
                s += [True]
            else:
                s += [False]
        S += [Solution(s, constraints)]                 #s =(59) a state of colored homes   S= (500)show efficinecy of each population
    return S
def crossover(P, constraints):
    #selection
    rules, nLines, nColumns, nPoints, nPopulation = constraints
    PP = []
    P = sorted(P, key=lambda s: (s.fitness, random.random()))
    n = (nPopulation * (nPopulation + 1)) / 2
    prob = [i / n for i in range(1, nPopulation + 1)]

    for _ in range(nPopulation):
        child1Points = []
        child2Points = []
        parent1, parent2 = random.choice(P, p=prob, replace=False, size=2)

        for i in range(nPoints):
            if random.random() <= 0.5:
                child1Points += [parent1.points[i]]
                child2Points += [parent2.points[i]]
            else:
                child1Points += [parent2.points[i]]
                child2Points += [parent1.points[i]]

        PP += [Solution(child1Points, constraints), Solution(child2Points, constraints)]

    return PP
def mutation(P, constraints, input_prob=(0.4/100)):
    rules, nLines, nColumns, nPoints, nPopulation = constraints

    PP = []

    for s in P:

        prob = input_prob
        if len(sys.argv) > 3:
            prob = float(sys.argv[3])

        newPoints = []

        for p in s.points:
            if random.random() > prob:
                newPoints += [p]
            else:
                newPoints += [not p]

        PP += [Solution(newPoints, constraints)]

    return PP
def select(P, PP, constraints,portionP , portionC):
    rules, nLines, nColumns, nPoints, nPopulation = constraints

    P = sorted(P, key=lambda s: (s.fitness, random.random()), reverse=True)
    PP = sorted(PP, key=lambda s: (s.fitness, random.random()), reverse=True)

    nParents = int(portionP * nPopulation) + 1
    nChildren = int(portionC * nPopulation) + 1
    nRandom = nPopulation - nChildren - nParents

    bestOnes = P[:nParents] + PP[:nChildren]
    others = P[nParents:] + PP[nChildren:]

    nextP = bestOnes + np.ndarray.tolist(random.choice(others, size=nRandom, replace=False))

    return nextP
def converge(P, constraints):
    rules, nLines, nColumns, nPoints, nPopulation = constraints

    for s in P:
        if s.fitness == 0:
            return True         #we reach answer

    for i in range(len(P) - 1):
        if P[i].points != P[i + 1].points:
            return False

    return True         #we reach equal population(not answer)
def best(P, constraints):
    rules, nLines, nColumns, nPoints, nPopulation = constraints

    for s in P:
        if s.fitness == 0:
            return s
    return P[0]
def population_test():
    populations = (100,300,500,700,900)
    dataset=("i11" , "i12")
    figs, axss = plt.subplots(2)
    for j in range(len(dataset)):
        number_of_iterations=np.zeros(len(populations))
        execution_time = np.zeros(len(populations))
        for i in range(len(populations)):
            print("Iteration number: %d" %i)
            t0 = time.time()
            number_of_iterations[i] = main(populations[i] , dataset[j])
            t1 = time.time()
            execution_time[i] = t1-t0
            # iterations=
            print("number of iteration is: %d \n" %number_of_iterations[i])
            print("time: %d \n\n" % execution_time[i])
        fig, axs = plt.subplots(2)
        fig.suptitle('Population Size Impact on: '+ dataset[j])
        axs[0].plot(populations , number_of_iterations)
        axs[0].set_ylabel("Numbers of Iterations. ")
        axs[0].set_xticks(populations)
        axs[0].grid()

        axs[1].plot(populations , execution_time)
        axs[1].set_xlabel("Numbers of Population.")
        axs[1].set_ylabel("Execution Time. ")
        axs[1].set_xticks(populations)
        axs[1].grid()

        axss[0].plot(populations, number_of_iterations)
        axss[1].plot(populations, execution_time)
        plt.savefig(dataset[j])



    figs.suptitle('Population Size Impact')
    axss[0].set_ylabel("Numbers of Iterations. ")
    axss[0].set_xticks(populations)
    axss[0].grid()
    axss[0].legend([dataset[0] , dataset[1]])

    axss[1].set_xlabel("Numbers of Population.")
    axss[1].set_ylabel("Execution Time. ")
    axss[1].set_xticks(populations)
    axss[1].grid()
    axss[1].legend([dataset[0], dataset[1]])
    plt.show()

def mutation_test():
    probs = (0.1/100, 0.3/100, 0.5/100, 0.7/100, 0.9/100)
    dataset = ("i11", "i12")
    figs, axss = plt.subplots(2)
    for j in range(len(dataset)):
        number_of_iterations = np.zeros(len(probs))
        execution_time = np.zeros(len(probs))
        for i in range(len(probs)):
            print("Iteration number: %d" % i)
            t0 = time.time()
            number_of_iterations[i] = main(500, dataset[j],probs[i])
            t1 = time.time()
            execution_time[i] = t1 - t0
            # iterations=
            print("number of iteration is: %d \n" % number_of_iterations[i])
            print("time: %d \n\n" % execution_time[i])
        fig, axs = plt.subplots(2)
        fig.suptitle('Probability Impact on: ' + dataset[j])
        axs[0].plot(probs, number_of_iterations)
        axs[0].set_ylabel("Numbers of Iterations. ")
        axs[0].set_xticks(probs)
        axs[0].grid()

        axs[1].plot(probs, execution_time)
        axs[1].set_xlabel("Probability")
        axs[1].set_ylabel("Execution Time. ")
        axs[1].set_xticks(probs)
        axs[1].grid()

        axss[0].plot(probs, number_of_iterations)
        axss[1].plot(probs, execution_time)
        plt.savefig(dataset[j])

    figs.suptitle('Probability Size Impact')
    axss[0].set_ylabel("Numbers of Iterations. ")
    axss[0].set_xticks(probs)
    axss[0].grid()
    axss[0].legend([dataset[0], dataset[1]])

    axss[1].set_xlabel("Probability.")
    axss[1].set_ylabel("Execution Time. ")
    axss[1].set_xticks(probs)
    axss[1].grid()
    axss[1].legend([dataset[0], dataset[1]])
    plt.show()

def gifted_test():
    portionP=(0.1 , 0.3 , 0.45 , 0.6 , 0.8)
    portionC=(0.8 , 0.6 , 0.45 , 0.3 , 0.1)
    dataset = ("i11", "i12")
    figs, axss = plt.subplots(2)
    for j in range(len(dataset)):
        number_of_iterations = np.zeros(len(portionP))
        execution_time = np.zeros(len(portionP))
        for i in range(len(portionP)):
            print("Iteration number: %d" % i)
            t0 = time.time()
            number_of_iterations[i] = main(500, dataset[j],0.4/100,portionP[i],portionC[i])
            t1 = time.time()
            execution_time[i] = t1 - t0
            # iterations=
            print("number of iteration is: %d \n" % number_of_iterations[i])
            print("time: %d \n\n" % execution_time[i])
        fig, axs = plt.subplots(2)
        fig.suptitle('Portion of Remaining Best Old Generation Impact on: ' + dataset[j])
        axs[0].plot(portionP, number_of_iterations)
        axs[0].set_ylabel("Numbers of Iterations. ")
        axs[0].set_xticks((0.1 , 0.3,0.45,0.6,0.8))
        axs[0].grid()

        axs[1].plot(portionP, execution_time)
        axs[1].set_xlabel("Portion")
        axs[1].set_ylabel("Execution Time. ")
        axs[1].set_xticks((0.1 , 0.3,0.45,0.6,0.8))
        axs[1].grid()

        axss[0].plot(portionP, number_of_iterations)
        axss[1].plot(portionP, execution_time)
        plt.savefig(dataset[j])

    figs.suptitle('ortion of Remaining Best Old Generation Impact on')
    axss[0].set_ylabel("Numbers of Iterations. ")
    axss[0].set_xticks((0.1, 0.3, 0.45, 0.6, 0.8))
    axss[0].grid()
    axss[0].legend([dataset[0], dataset[1]])

    axss[1].set_xlabel("Portion.")
    axss[1].set_ylabel("Execution Time. ")
    axss[1].set_xticks((0.1, 0.3, 0.45, 0.6, 0.8))
    axss[1].grid()
    axss[1].legend([dataset[0], dataset[1]])
    plt.show()


if __name__ == '__main__':
        main()
        # population_test()
        # mutation_test()
        # gifted_test()