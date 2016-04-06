# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 23:27:51 2015

@authors
Hamsavardhini Deventhiran
Prabhu Ramamurthy

Reference-link : http://www.omgwiki.org/hpec/files/hpec-challenge/ga.html

"""

import pandas as pd
from operator import add, itemgetter
import random


# Machine[M1,M2,M3] - Product[P1,P2,P3,P4,P5] matrix (BOM)
# 1 => Product[p] undergoes Machine[M] process for manufacturing 
# 0 => Product[p] doesn't undergoes Machine[M] process for manufacturing  
build_of_material =[[1,1,0,1,1],[1,0,0,0,1],[0,1,0,1,0]]

# Machine[M1,M2,M3] - thresold Load capacity per month beyond which the machine can't process
machine_capacity = [4200,3500,4000]

# Product[P1,P2,P3,P4,P5] - individual selling unit price
product_selling_price = [5,8,3,10,2]

# Total Cost expenditure per month inclusive of manufacturing cost,Labour cost,etc
total_expense = 9000  


# Process: Reading inital population from Excel file
# Input : file path
# Output : Product List created from the input file
def readInitialPopulation ():
    #####print "Reading Data.."
    data = pd.read_csv(file_path)
    # Creating the list from input
    product_list = [] 
    for i,t in data.iterrows():
        product_list.append([t.Product1,t.Product2,t.Product3,t.Product4,t.Product5])
    # Saving the maximum and minium counts for individual products from demand list of current month plan
    # The demand plan scenrios vary very month and the value will have certain business bounds
    # Helps in controlling the mutation activity (boundary mutation type) - mutating the gene value within controlled limits
    # Saving as global values - used for mutation functionality  
    global min_quantity, max_quantity
    min_quantity, max_quantity =[],[]    
    
    i=0
    while i < (len(data.columns)-1):
        min_quantity.append(reduce(min,[row[i] for row in product_list]))
        max_quantity.append(reduce(max,[row[i] for row in product_list]))
        i+=1
    return product_list



# Process: Each demand-scenario(chromosome) will be evaluated for its profit contribution 
#          only if BOM constraints satisfied
# Input : a demand-scenario 
# Output : demand-scenario's profit contribution
def evaluateChromosome (scenario):
    #####print "Evaluate chromosome.."
    mach_capacity = []
    i=0
    while i < len(build_of_material):
        mach_capacity.append(reduce(add,[pl*m for pl,m in zip(scenario,build_of_material[i])],0))
        i=i+1
    # The chromosome that doesn't suit the BOM constraint gets rejected     
    if (machine_capacity[0]>=  mach_capacity[0] and machine_capacity[1]>=mach_capacity[1] and machine_capacity[2]>=  mach_capacity[2]):
        # Calculating the selling price, revenue for BOM constraint satisfied chromosomes        
        selling_price = [sc*pf for sc,pf in zip(scenario,product_selling_price)]
        revenue = reduce(add, selling_price,0)
        profit = float(revenue - total_expense)/float(total_expense)*100
        # Ignoring the chromosomes that results in loss (negative profit)
        if profit>0:
            return profit
        else:
            return 0 
    else:
        return 0
 
 
# Process: chromosome is assigned with a fitness score to handle chromosome-selection and gene-replacement 
#          strategies if required      
# Input : Errors List of all demand scenarios
# Output : Fitness scores List for all demand scenarios
def calculateFitness (errors):
    #####print "Calculate Fintess.."
    fitness_scores = []
    # Total fitness of that generation population - for roulette wheel chromosome selection method
    # Helps in shifting the 
    total_fitness_score= reduce (add,errors,0) 
    print '-Average Fitness Score := ',(total_fitness_score/len(errors))
    i = 0
    for error in errors:
        # Probability of selection for each chromsome - for roulette wheel chromosome selection method
        fitness_scores.append(float(errors[i])/float(total_fitness_score))
        i += 1
    return fitness_scores


# Process: Ranking the demand-scenario(chromosome) based upon the profit contribution
# Inputs : Demand Scenarios List, Targetted profit value
# Output : Demand Scenario List sorted with fitness scores  
def rankPopulation (scenarios):
    #####print "Rank Population.."
    fitness_scores,errors = [],[]
    # Evaluating each chromosome for given target and if the minimal target is achieved that particular chromsome gets selected
    i=1
    for scenario in scenarios:
        scenario_profit = evaluateChromosome(scenario)
        error = (scenario_profit - target_profit)
        # Setting a global indicator to notify - solution is found
        if error > 0: 
            global solution_found
            solution_found = True
            print '-SOLUTION OBTAINED IN THIS GENERATION'
        errors.append(error)
        i+=1
    # Calculating the fitness of chromsomes in current population    
    fitness_scores = calculateFitness(errors)
    # Grouping and Ranking the chromsomes with their fitness-scores 
    unranked_population = zip (scenarios,errors,fitness_scores)
    ranked_population = sorted (unranked_population,key = itemgetter(-1), reverse = False )
    # Chromosome with highest fitness-score tops the ranked population list
    return ranked_population
    
    
# Process : Selecting the fittest parent for breeding using roulette wheel method
# Inputs : Parent chromosome list , their fitness-scores list
# Output : 2 best performing parent chromosomes    
def selectFittest (fitness_scores, ranked_chromosome):
    # Ensuring that 2 different parent chromosomes are selected (here - index of parents is used for difference criteria)
    while 1 == 1: 
        index1 = rouletteWheel (fitness_scores)
        index2 = rouletteWheel (fitness_scores)
        if index1 == index2:
            continue
        else:
            break
    # Selecting the Parent chromsomes
    ch1 = ranked_chromosome[index1]  
    ch2 = ranked_chromosome[index2]
    return ch1, ch2


# Process : Roulette wheel - Fitness proportionate selection method
# Input : Fitness score list
# Output : Index value of a parent chromosome
def rouletteWheel (fitness_scores):
    #####print "Roulette.."
    cumalative_fitness = 0.0
    r = random.random()
    for i in range(len(fitness_scores)): 
        # Cummulative propability of each chromosome
        cumalative_fitness += fitness_scores[i]
        if cumalative_fitness > r:
            return i  
 
 
# Process : Breeding Parent Chromosomes - using Crossover or Mutation methods
# Inputs : 2 parent chromosomes, crossover-rate, mutation-rate
# Outputs : Newly generated 2 child chromosomes
def breedChromosomes (ch1, ch2):
    #####print "Breeding.."
    newCh1, newCh2 = [], []
    # Executing crossover if random number generated is less than crossover-rate
    # Hint - Maximum the crossover rate => higher the chances of crossover occurences
    if random.random() < crossover_rate:
        # Crossing over based upon the limitations of no.of.genes in each chromosome 
        r=random.randint(1,crossover_genesper_chormosome)
        newCh1, newCh2 = crossover(ch1, ch2,r)
        return newCh1, newCh2
    else:
        newCh1, newCh2 = ch1, ch2
        # If Crossing over not happens, then mutation happens
        newnewCh1 = mutate (newCh1)
        newnewCh2 = mutate (newCh2)
        return newnewCh1, newnewCh2
        
        
# Process : Single-Point Crossover method
# Inputs : Parent chromosome, crossover-point
# Outputs : Child chromsomes afte crossover
def crossover (ch1, ch2, cp):
    #####print "Crossover.."
    return ch1[:cp]+ch2[cp:], ch2[:cp]+ch1[cp:]

# Process : Boundary Mutation method - to maintain genetic diversity within the specified limits
# Inputs : A parent chromosome, mutation-rate
# Output : single mutated child chromosome  
def mutate (ch):
    #####print "Mutate.."
    mutatedCh = []
    for i in ch:
        # Hint - Minimize the mutation rate => lesse the chances of mutation occurences
        if random.random() < mutation_rate: 
                mutatedCh.append(random.randint(min_quantity[ch.index(i)],max_quantity[ch.index(i)])) 
        else:
            #Hint - at times,parent chromosome is forwarded to next generation without mutating 
            mutatedCh.append(i)
    return mutatedCh


# Process: Generating a new population (chromosomes set) with through Elitism,Crossover,Mutation
# Input : Fitness score sorted current generation population 
# Output : Next generation population  
def nextGenerationPopulation (ranked_population):
    #####print "Generate Population"
    new_population = []
    # Fixing 1.5 % incremented population size for next genration
    new_population_size = int(len(ranked_population)*population_increase_rate) 
    # Extract fitness scores and chromosomes from ranked population
    fitness_scores = [ item[-1] for item in ranked_population ] 
    ranked_chromosome = [ item[0] for item in ranked_population ] 
    # Elitism (conserving the best few chromosomes to new population)
    elitism_population_size = 2
    # Conserving top 2 parent chromosomes to next generation
    new_population.extend(ranked_chromosome[:elitism_population_size]) 
    # Selecting the parent chromsosme in random and Breeding them to create child chromsome 
    while len(new_population) < new_population_size:
        ch1, ch2 = [], []
        ch1, ch2 = selectFittest (fitness_scores, ranked_chromosome)
        ch1, ch2 = breedChromosomes (ch1, ch2)  
        new_population.append(ch1) 
        new_population.append(ch2)
    return new_population
    
#### TESTING

target_profit = 50 # Value in percentage (range 1-100)
crossover_rate  = 0.8 # Value in percentage (range 0-1, hint:higher the better, for best results >=0.6)
mutation_rate = 0.2 # Value in percentage (range 0-1, hint:lower the better, for best results <=0.2)
crossover_genesper_chormosome = 4 # Chromosome gene count (range 1-4)
population_increase_rate =1 # Value in count (range 0.9-1, hint:to control Generation-Gap, for best results use 1)
file_path = "/path/to/mock-data csv file" # File Location

iterations = 1
max_iterations = 500
solution_found = False
chromos = readInitialPopulation() #generate new population of random chromosomes
print '\nInitial Population:', chromos

while iterations != max_iterations:# take the pop of random chromos and rank them based on their fitness score/proximity to target output
    print '\nGeneration #:', iterations
    rankedPop = rankPopulation(chromos)
    print '-Chromosome List := ', [item[0] for item in rankedPop]
    print '-Population Size := ', len([item[0] for item in rankedPop])
    if solution_found != True:
        # if solution is not found iterate a new population from previous ranked population
        chromos = []
        chromos = nextGenerationPopulation(rankedPop)
        iterations += 1
    else:
        print '\n******SOLUTION******'
        print "Demand Scenario = ", [item[0] for item in rankedPop].pop(0)
        print "Profit %  = ", target_profit + [item[1] for item in rankedPop].pop(0)
        print '********************'
        break

#######  
  




 