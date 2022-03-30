############################################################
#   Dev: Josias Moukpe
#   Class: Machine Learning
#   Date: 2/22/2022
#   file: genetic.py
#   Description: : main genetic algorithm file
#############################################################

from utils import lg
from random import randint, uniform, sample

class Genetic:
    '''Main class for Genetic Algorithm'''


    def __init__(
        self, 
        training, 
        testing,
        attributes,
        population,
        mutation,
        replacement,
        max_generations,
        fitness_threshold,
        selection_type,
        debug
        ) -> None:
        
        '''Initialize the Genetic Algorithm'''

        # read params
        # self.training = training
        # self.testing = testing
        # self.attributes = attributes
        self.population_size = population
        self.mutation_rate = mutation
        self.replacement_rate = replacement
        self.max_generations = max_generations
        self.fitness_threshold = fitness_threshold
        self.selection_type = selection_type
        self.debug = debug

        # read the data
        self.attributes, self.inputs, self.outputs = self.read_attributes(attributes)
        self.training = self.read_data(training)
        self.testing = self.read_data(testing)

        # if self.debug:
        #     print('Attributes: ', self.attributes)
        #     print('Order: ', self.order)
        #     print('Final Attribute: ', self.order[-1])

        # determine single rule length
        self.rule_length = 0
        # getting the precondition length
        for attr in self.inputs:
            self.rule_length += len(self.attributes[attr])
        # getting the postcondition length
        for attr in self.outputs:
            self.rule_length += int(lg(len(self.attributes[attr])))

        # max number of rules (should not be greater than 
        # number of examples present in the training set)
        self.FACTOR = 2 # scaling factor to determine the max number of rules
        self.ruleset_length = len(self.training) // self.FACTOR

        if self.debug:
            print('Rule Length: ', self.rule_length)
            print('Rules Max Count: ', self.ruleset_length)

        # instanting params for the GA
        self.population = []
        self.fitness = []
        self.best_individual = None

        # generate the population
        self.generate_population()


    def __repr__(self) -> str:
        '''Returns a string representation of the GA'''
        res = 'Genetic Algorithm\n'
        res += '- Mutation: {}\n'.format(self.mutation_rate)
        res += '- Replacement: {}\n'.format(self.replacement_rate)
        res += '- Max Generations: {}\n'.format(self.max_generations)
        res += '- Fitness Threshold: {}\n'.format(self.fitness_threshold)
        res += '- Selection Type: {}\n'.format(self.selection_type)
        res += '- Population Size: {}\n'.format(self.population_size)
        res += '- Population: {}\n'.format(self.population)
        
        return res


    def read_data(self, path):
        '''Reads data from a file'''
        
        data = []

        # read in the attributes
        with open(path, 'r') as f:
            for line in f:
                    words = line.strip().split()
                    data.append(words)
               
        if self.debug:
            print('Read data: ', data)

        if len(data) == 0:
            raise Exception('No data found')

        return data

    def read_attributes(self, path):
        '''Reads attributes from a file'''
        attributes = {}
        inputs, outputs = [], []
        is_input = True

         # read in the attributes
        with open(path , 'r') as f:
            for line in f:
                if len(line) > 1:
                    words = line.strip().split()
                    
                    # storing the attributes
                    attributes[words[0]] = words[1:]


                    # storing the inputs and outputs
                    if is_input:
                        inputs.append(words[0])
                    else:
                        outputs.append(words[0])

                else:
                    is_input = False
          
        if self.debug:
            print('Attributes: ', attributes)
            print('Inputs: ', inputs)
            print('Outputs: ', outputs)

        if len(attributes) == 0:
            raise Exception('No attributes found')

        return attributes, inputs, outputs

    def generate_individual(self):
        '''Generates and return individual at random
        '''
        # generate random length of individual
        individual_len = randint(1, self.ruleset_length) * self.rule_length
        # generate random individual
        individual = ""
        for _ in range(individual_len):
            individual += str(randint(0, 1))

        return individual

    def generate_population(self):
        '''Generates a population of individuals'''

        # generate a population of individuals
        for _ in range(self.population_size):
            self.population.append(self.generate_individual())

        if self.debug:
            print('Population: ', self.population)


    def crossover(self, parent1, parent2):
        '''Crossover between two parents'''
        pass

    def mutation(self):
        '''Mutation of individuals at random
            single bit mutation
        '''
        # choose population to mutate
        num_mutants = int(self.mutation_rate * self.population_size)
        # sample the population of mutants the index of the mutants
        mutants = sample(range(self.population_size), num_mutants)
        # mutate the population
        for mutant in mutants:
            # choose a random position to mutate
            bit = randint(0, len(self.population[mutant]) - 1)
            # flip the bit
            self.population[mutant] = self.population[mutant][:bit] + \
                str(1 - int(self.population[mutant][bit])) + \
                self.population[mutant][bit + 1:]

        # if self.debug:
        #     print('Mutated Population: ', self.population)

        

    # TODO: implement binary decoding of rules
    def decode(self, individual):
        '''Decodes an individual'''
        pass

    def evaluate_rules(self, test_data):
        '''Evaluate the rules on test data'''
        pass 

    def fitness(self, individual):
        '''Measures Fitness of an individual'''
        pass

    def selection(self, population, fitness):
        '''Selection of parents based on the type 
        of selection chosen'''
        pass

    def run(self):
        '''Run the Genetic Algorithm'''
        
        # initialize the population
        self.generate_population()

        # # run the GA
        # for _ in range(self.max_generations):
        #     # evaluate the population
        #     self.evaluate()
        #     # select the best individual
        #     self.select()
        #     # crossover
        #     self.crossover()
        #     # mutate
        #     self.mutate()
        #     # replace the worst individuals
        #     self.replace()

        # # evaluate the population
        # self.evaluate()
        # # select the best individual
        # self.select()

        # # print the best individual
        # print('Best Individual: ', self.best_individual)


    def test(self, individual, data=None):
        '''Tests an individual on a dataset'''
        
        data = data or self.testing


    # TODO: implement to support continuous attributes
    def discretize(self, data):
        '''Discretizes the data'''
        pass