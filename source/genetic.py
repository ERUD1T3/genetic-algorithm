############################################################
#   Dev: Josias Moukpe
#   Class: Machine Learning
#   Date: 2/22/2022
#   file: genetic.py
#   Description: : main genetic algorithm file
#############################################################

from distutils.log import debug
from utils import lg
from random import randint, choices, sample

class Genetic:
    '''Main class for Genetic Algorithm'''

    def __init__(
        self, 
        training, 
        testing,
        attributes,
        size,
        mutation,
        replacement,
        max_generations,
        threshold,
        selection_type,
        debug
        ) -> None:
        
        '''Initialize the Genetic Algorithm'''

        # read params
        # self.training = training
        # self.testing = testing
        # self.attributes = attributes
        # self.population_size = population
        self.mutation_rate = mutation
        self.replace_rate = replacement
        self.max_generations = max_generations
        self.threshold = threshold
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
            self.rule_length += round(lg(len(self.attributes[attr])))

        # max number of rules (should not be greater than 
        # number of examples present in the training set)
        self.FACTOR = 2 # scaling factor to determine the max number of rules
        self.ruleset_length = len(self.training) // self.FACTOR

        if self.debug:
            print('Rule Length: ', self.rule_length)
            print('Rules Max Count: ', self.ruleset_length)

        # instanting params for the GA
        self.population = [] # population
        self.fitnesses = [] # fitnesses of individuals
        self.probs = [] # probabilities of individuals
        self.best = None # index, value, fitness of best individual

        # generate the population
        self.generate_population(size)


    def __repr__(self) -> str:
        '''Returns a string representation of the GA'''
        res = 'Genetic Algorithm\n'
        res += '- Mutation: {}\n'.format(self.mutation_rate)
        res += '- Replacement: {}\n'.format(self.replace_rate)
        res += '- Max Generations: {}\n'.format(self.max_generations)
        res += '- Fitness Threshold: {}\n'.format(self.threshold)
        res += '- Selection Type: {}\n'.format(self.selection_type)
        res += '- Population Size: {}\n'.format(len(self.population))
        res += '- Population: {}\n'.format(self.population)
        
        return res

    def display_population(self):
        '''Displays the population'''
        print('Population: ', self.population)

    def encode_data(self, instance)->str:
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
            num_values = round(lg(len(self.attributes[attribute])))
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

                    # if self.debug:
                    #     print('Words: ', words)
                    #     print('Encoded: ', encoded)

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

    def generate_population(self, size):
        '''Generates a population of individuals'''

        # generate a population of individuals
        for _ in range(size):
            self.population.append(self.generate_individual())

        if self.debug:
            print('Population: ', self.population)

    def mutate(self, population):
        '''Mutation of individuals at random
            single bit mutation
        '''
        # get the population size
        pop_size = len(population)

        # choose population to mutate
        num_mutants = round(self.mutation_rate * pop_size)
        # sample the population of mutants the index of the mutants
        mutants = sample(range(pop_size), num_mutants)
        # mutate the population
        for mutant in mutants:
            # choose a random position to mutate
            bit = randint(0, len(population[mutant]) - 1)
            # flip the bit
            population[mutant] = population[mutant][:bit] + \
                str(1 - int(population[mutant][bit])) + \
                population[mutant][bit + 1:]

        # if self.debug:
        #     print('Mutated Population: ', population)

        return population
    
    def and_operator(self, ante1, ante2):
        '''AND operator for antecedents'''

        res = ""
        for i in range(len(ante1)):
            res += '1' if ante1[i] == '1' and \
                 ante2[i] == '1' else '0'

        return res

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
            if len(votes) > 0:
                most_voted = max(votes, key=votes.get)
                # if self.debug:
                #     print('Votes: ', votes)
                #     print('Most voted: ', most_voted)
                return most_voted
            else:
                return None

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
            class_example = example[self.ante_length:]
            # get the class of the individual
            class_individual = self.classify(individual, example)
            if class_individual == class_example:
                corrects += 1
            else:
                incorrects += 1

        accuracy = corrects / len(data)
        # if self.debug:
        #     print('Corrects: ', corrects)
        #     print('Incorrects: ', incorrects)
        #     print('Accuracy: ', accuracy)
        
        return accuracy

    def fitness(self, individual):
        '''Measures Fitness of an individual'''
            
        # get the accuracy of the individual
        accuracy = self.test_accuracy(individual)
        # return the fitness
        return accuracy ** 2

    def evaluate(self):
        '''Evaluates the population'''

        # evaluate the population
        if len(self.fitnesses) == 0:
            for individual in self.population:
                self.fitnesses.append(self.fitness(individual))
        else:
            size = len(self.population)
            for i in range(size):
                self.fitnesses[i] = self.fitness(self.population[i])

        # save the index, value, and fitness of best individual
        best_fitness = max(self.fitnesses)
        if self.best is None or best_fitness > self.best[2]:
            best_index = self.fitnesses.index(best_fitness)
            self.best = (best_index, self.population[best_index], best_fitness)

        if self.debug:
            print('Fitnesses of Population: ', self.fitnesses)
            print('Best Individual: ', self.best)

        return self.best
    
    
    # TODO: test this function
    def generate_crossover_pts(self, parent, d1=None, d2=None):
        '''Generates crossover points
            CAN BE IMPROVED FOR EFFICIENCY
        '''
        # get the length of the parent
        upper_bound = len(parent) - 1
        # get the crossover points
        if d1 is None and d2 is None:
            cpt1 = randint(0, upper_bound)
            cpt2 = randint(0, upper_bound)
            while cpt2 == cpt1:
                cpt2 = randint(0, upper_bound)

            # get distances d1 and d2
            leftmost = min(cpt1, cpt2)
            rightmost = max(cpt1, cpt2)
            d1 = leftmost % self.rule_length
            d2 = rightmost % self.rule_length

            return cpt1, cpt2, d1, d2
        # get the crossover points matching d1 and d2
        else:
            d3, d4 = None, None
            while (d3 is None and d4 is None) or \
                (d3 != d1 or d4 != d2):
                # get first random crossover points in parent2
                cpt3 = randint(0, upper_bound)
                # get second random crossover points in parent2
                cpt4 = randint(0, upper_bound)
                while cpt4 == cpt3:
                    cpt4 = randint(0, upper_bound)

                # get distance d3 and d4
                leftmost = min(cpt3, cpt4)
                rightmost = max(cpt3, cpt4)
                d3 = leftmost % self.rule_length
                d4 = rightmost % self.rule_length

            return cpt3, cpt4, d3, d4


    # TODO: test this function
    def crossover_op(self, parent1, parent2):
        '''Crossover between two parents rules to generate
           two children of variable length'''
        
        # get the crossover points of the parents
        cpt1, cpt2, d1, d2 = self.generate_crossover_pts(parent1)
        cpt3, cpt4, _, _ = self.generate_crossover_pts(parent2, d1, d2)

        # get the children (check if the crossover points are valid)
        child1 = parent1[:cpt1] + parent2[cpt3:cpt4] + parent1[cpt2:]
        child2 = parent2[:cpt3] + parent1[cpt1:cpt2] + parent2[cpt4:]
        # child1 = parent1[:cpt1] + parent2[cpt3:cpt4] + parent1[cpt1:cpt2] + parent2[cpt2:]
        # child2 = parent2[:cpt3] + parent1[cpt1:cpt2] + parent2[cpt3:cpt4] + parent1[cpt2:]

        return child1, child2

    def get_probs(self):
        '''Returns the probabilities of each individual'''

        if len(self.fitnesses) == 0:
            self.evaluate()

        if len(self.probs) == 0:
            # get the sum of fitnesses
            sum_fitnesses = sum(self.fitnesses)
            # get the probabilities
            probs = [f / sum_fitnesses for f in self.fitnesses]
            
            self.probs = probs

            return probs
        else:
            return self.probs

    # TODO: test this function
    def crossover(self):
        '''Crossover between parents to generate children according to probability'''
        # get the population size
        pop_size = len(self.population)
        # get number of individuals to crossover
        
        num_pairs = round(self.replace_rate * pop_size) // 2
        
        # get the probabilities
        probs = self.get_probs()
        # get the pairs of individuals to crossover
        pairs = choices(range(pop_size), probs, k= (num_pairs * 2))
        # pairs = sample(range(pop_size), num_pairs * 2)
        # get the children
        children = []
        for i in range(0, len(pairs), 2):
            *twins, = self.crossover_op(
                self.population[pairs[i]], 
                self.population[pairs[i + 1]]
            )
            children += twins

        # add the children to the population
        # self.population += children
        return children

    # TODO: implement
    def tournament_selection(self):
        '''Tournament selection of survivors'''
        pass

    # TODO: implement
    def rank_selection(self, population):
        '''Rank selection of survivors'''
        pass

    # TODO: test
    def proportional_selection(self, population):
        '''Fitness proportional selection of parents'''
       # get the population size
        size = len(population)
        # get the number of survivors to select
        k_survivors = round((1 - self.replace_rate) * size)
        # get probabilities
        probs = self.get_probs()

        if self.debug:
            print('num of probs: ', len(probs))
            print('num of survivors: ', size)
            print('stored size: ', len(self.population))

        # get the survivors 
        survivors = choices(population, probs, k=k_survivors)
        
        return survivors

    # TODO: test
    def select(self):
        '''Selection of parents based on the type 
        of selection chosen'''
        
        if self.selection_type == 'P':
            return self.proportional_selection(self.population)
        elif self.selection_type == 'T':
            return self.tournament_selection(self.population)
        elif self.selection_type == 'R':
            return self.rank_selection(self.population)
        else:
            raise ValueError('Invalid Selection Type')

    # TODO: test    
    def run(self):
        '''Run the Genetic Algorithm'''
        
        # initialize the population
        # self.generate_population(self.size )
        # evaluate the population
        self.evaluate()
        # print the best individual
        # print('Best Individual: ', self.best)
        for g in range(self.max_generations):
            if self.debug:
                print('Generation: ', g)
            new_population = []
            # get the survivors
            survivors = self.select()
            new_population += survivors
            # crossover
            children = self.crossover()
            new_population += children
            # mutate
            self.mutate(new_population)
            # update
            # check they are same size
            if self.debug:
                print('new size: ', len(new_population))
                print('old size: ', len(self.population))
            if len(new_population) != len(self.population):
                raise ValueError('Population size mismatch')
            self.population = new_population
            # evaluate
            self.evaluate()
            # stop early if threshold is reached
            if self.best[2] >= self.threshold:
                if self.debug:
                    print('Threshold Reached')
                break
    
        # print the best individual
        print('Best Individual: ', self.best)
        self.print_individial(self.best)
        # return the best individual
        return self.best

    def decode_rule(self, rule):
        '''Decode the rule binary to readable format'''
        res = ''
        # get values from rule
        for attr in self.inputs:
            # get number of values
            num_values = len(self.attributes[attr])
            # get the value substring
            value = rule[:num_values]
            if self.debug:
                print(f'{attr}: {value}')
            # check if value is 0s
            if value != '0' * num_values:
                res += f'{attr} = ('
                # get the attribute value
                for i in range(num_values):
                    if value[i] == '1':
                        res += f'{self.attributes[attr][i]} v '
                # remove the last 'v '
                res = res[:-3]
                res += ') ^ '
            # get rest of the rule
            rule = rule[num_values:]

        # remove the last '^ '
        if res != '': res = res[:-3]
        res += ' => '

        # get the output value
        for attr in self.outputs:
            # get number of values
            num_values = round(lg(len(self.attributes[attr])))
            # get the value substring
            value = rule[:num_values]
            # if self.debug:
            #     print(f'{attr}: {value}')
            res += f'{attr} = ('
            # convert the value to decimal
            index = int('0b'+ value, 2)
            # get the attribute value
            res += f'{self.attributes[attr][index]}'
            res += ') ^ '
            # get rest of the rule
            rule = rule[num_values:]

        # remove the last '^ '
        res = res[:-3]
        res += '\n'

        return res

    def print_individial(self, individual):
        '''print an individual bit string'''
        # get rules 
        for r in range(0, len(individual), self.rule_length):
            rule = individual[r:r + self.rule_length]
            # decode the rule
            rule = self.decode_rule(rule)
            # print the rule
            print(rule, end='')

    def test(self, data):
        '''Test the algorithm by testing best individual'''
        # get the best individual
        if self.best is None:
            self.run()
        # get accuracy of the best individual
        accuracy = self.test_accuracy(self.best[1], data)
        # print the accuracy
        print(f'Accuracy: {accuracy * 100}%')
            
    # TODO: implement to support continuous attributes
    def discretize(self, data):
        '''Discretizes the data'''
        pass