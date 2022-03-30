############################################################
#   Dev: Josias Moukpe
#   Class: Machine Learning
#   Date: 2/22/2022
#   file: genetic.py
#   Description: : main genetic algorithm file
#############################################################

from base64 import encode
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
        
        # save the precondition length
        self.ante_length = self.rule_length

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
        self.fitnesses = []
        self.best = None

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

    def encode_data(self, instance: list[str])->str:
        '''Encodes the instance'''
        encoded = ''

        # encode the inputs
        for v in range(len(self.inputs)):
            tmp = ''
            # get value
            value = instance[v]
            # get attribute
            attribute = self.inputs[v]
            # get number of values
            num_values = len(self.attributes[attribute])
            # get index of value in attribute
            index = self.attributes[attribute].index(value)
            # set the bit of the index to 1 and rest to 0
            for i in range(num_values):
                tmp += '1' if i == index else '0'

            # add tmp to encoded
            encoded += tmp
        
        # encode the outputs
        for v in range(len(self.inputs), len(instance)):
            tmp = ''
            # get value
            value = instance[v]
            # get attribute
            attribute = self.outputs[v - len(self.inputs)]
            # get number of values
            num_values = int(lg(len(self.attributes[attribute])))
            # get index of value in attribute
            index = self.attributes[attribute].index(value)
            # get the binary representation of the index
            tmp = bin(index)[2:].zfill(num_values)
            # add tmp to encoded
            encoded += tmp

        return encoded

        


    def read_data(self, path):
        '''Reads data from a file'''
        
        data = []

        # read in the attributes
        with open(path, 'r') as f:
            for line in f:
                    words = line.strip().split()
                    encoded = self.encode_data(words)

                    if self.debug:
                        print('Words: ', words)
                        print('Encoded: ', encoded)
                        
                    data.append(encoded)
               
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


    # TODO: implement crossover
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
    

    # TODO: test this function
    def and_operator(self, ante1, ante2):
        '''AND operator for antecedents'''

        res = ""
        for i in range(len(ante1)):
            res += '1' if ante1[i] == '1' and \
                 ante2[i] == '1' else '0'

        return res

    # TODO: test this function 
    def rule_classify(self, rule, example):
        '''Evaluates a rule on training example'''

        # get the rule antecedent
        ante_r = rule[:self.ante_length]
        ante_e = example[:self.ante_length]

        # check if example satisfies the rule
        if self.and_operator(ante_r, ante_e) == ante_e:
            # get the rule consequent
            cons_r = rule[self.ante_length:]
            return cons_r
        else:
            return None

    # TODO: test this function
    def classify(self, individual, example, voting=True):
        '''use an individual to classify the example using voting'''

        if voting:
            votes = {}
            # split individual into rules
            num_rules = len(individual) // self.rule_length
            for r in range(num_rules):
                # get the rule
                rule = individual[r * self.rule_length: (r + 1) * self.rule_length]
                # classify the example
                res = self.rule_classify(rule, example)

                if res is not None:
                    if res in votes:
                        votes[res] += 1
                    else:
                        votes[res] = 1
                
            # get the most voted class
            most_voted = max(votes, key=votes.get)
            if self.debug:
                print('Votes: ', votes)
                print('Most voted: ', most_voted)
            
            return most_voted

        else:
            # split individual into rules
            num_rules = len(individual) // self.rule_length
            for r in range(num_rules):
                # get the rule
                rule = individual[r * self.rule_length: (r + 1) * self.rule_length]
                # classify the example
                res = self.rule_classify(rule, example)

                if res is not None:
                    return res
            
            return None

    def test_accuracy(self, individual, data=None):
        '''Tests an individual on a dataset'''
        
        data = data or self.training
        corrects, incorrects = 0, 0
        for example in data:
            # get the class of the example
            class_example = example[-1]
            # get the class of the individual
            class_individual = self.classify(individual, example)
            if class_individual == class_example:
                corrects += 1
            else:
                incorrects += 1

            
    # TODO: implement to support continuous attributes
    def discretize(self, data):
        '''Discretizes the data'''
        pass