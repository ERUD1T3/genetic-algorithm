############################################################
#   Dev: Josias Moukpe
#   Class: Machine Learning
#   Date: 2/22/2022
#   file: genetic.py
#   Description: : main genetic algorithm file
#############################################################

from unicodedata import decimal
from utils import lg
from random import randint, choice, choices, sample

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
        # check if dataset is Iris
        if 'Iris' in self.attributes:
            # getting the precondition length
            # 4 attributes represented by 2 float boundaries 
            # each encoded as 6 bits using decimal to bin encoding
            # with 3 bits for integer part and 3 bits for decimal part 
            self.iris = True
            self.int_len, self.dec_len = 3, 3
            self.bin_len = self.int_len + self.dec_len
            self.ante_len = 2 * self.bin_len * len(self.inputs) 
            self.rule_len = self.ante_len
            # getting the postcondition length
            for attr in self.outputs:
                self.rule_len += round(lg(len(self.attributes[attr])))

            if self.debug:
                print('Rule Length: ', self.rule_len)
                print('Precondition Length: ', self.ante_len)
                print('Postcondition Length: ', self.rule_len - self.ante_len)
        else:
            self.iris = False
            # determine single rule length
            self.rule_len = 0
            # getting the precondition length
            for attr in self.inputs:
                self.rule_len += len(self.attributes[attr])
            # save the precondition length
            self.ante_len = self.rule_len
            # getting the postcondition length
            for attr in self.outputs:
                self.rule_len += round(lg(len(self.attributes[attr])))

        self.training = self.read_data(training)
        self.testing = self.read_data(testing)

        # max number of rules (should not be greater than 
        # number of examples present in the training set)
        self.FACTOR = .5 # scaling factor to determine the max number of rules
        self.ruleset_len = round(len(self.training) * self.FACTOR)

        if self.debug:
            print('Rule Length: ', self.rule_len)
            print('Rules Max Count: ', self.ruleset_len)

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

    def encode_data(self, instance, just_output=False)->str:
        '''Encodes the instance'''
        encoded = ''
        in_part = instance[:len(self.inputs)]
        out_part = instance[len(self.inputs):]

        if just_output:
            # just encode output
            out_part = instance
        else:
            # encode the inputs
            for v in range(len(in_part)):
                tmp = ''
                # get value
                value = in_part[v]
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
        for v in range(len(out_part)):
            tmp = ''
            # get value
            value = out_part[v]
            # get attribute
            attribute = self.outputs[v]
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
                    # check if dataset is Iris
                    if not self.iris:
                        # convert words to floats
                        words = self.encode_data(words)
                        # if self.debug:
                        #     print('Words: ', words)
                        #     print('Encoded: ', encoded)
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
                # about to ready the output atributes
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
        '''
        Generates and return individual at random
        '''
        # generate random length of individual
        individual_len = randint(1, self.ruleset_len) * self.rule_len
        # generate random individual
        individual = ""
        for _ in range(individual_len):
            individual += str(randint(0, 1))
        # return the individual
        return individual

    def generate_population(self, size):
        '''Generates a population of individuals'''
        # generate a population of individuals
        for _ in range(size):
            individual = self.generate_individual()
            # check if the individual is valid
            while not self.is_valid(individual):
                individual = self.generate_individual()
            # add valid individual to population
            self.population.append(individual)
        # print the population
        if self.debug: print('Population: ', self.population)

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
        # check if dataset is Iris
        if self.iris:
            # get the rule antecedent
            ante_r = rule[:self.ante_len]
            ante_e = example[:len(self.inputs)]
            # check if example is covered by rule
            if self.is_match(ante_r, ante_e):
                # get the rule consequent
                cons_r = rule[self.ante_len:]
                return cons_r
            else:
                return None
        else:
            # get the rule antecedent
            ante_r = rule[:self.ante_len]
            ante_e = example[:self.ante_len]

            # check if example satisfies the rule
            if self.and_operator(ante_r, ante_e) == ante_e:
                # get the rule consequent
                cons_r = rule[self.ante_len:]
                return cons_r
            else:
                return None

    def classify(self, individual, example, voting=True):
        '''use an individual to classify the example using voting'''

        if voting:
            votes = {}
            # split individual into rules
            num_rules = len(individual) // self.rule_len
            for r in range(num_rules):
                # get the rule
                rule = individual[r * self.rule_len: (r + 1) * self.rule_len]
                # classify the example
                res = self.rule_classify(rule, example)
                # collect the votes
                if res is not None:
                    if res in votes: votes[res] += 1
                    else: votes[res] = 1
            # get the most voted class
            if len(votes) > 0:
                most_voted = max(votes, key=votes.get)
                return most_voted
            else:
                return None

        else:
            # split individual into rules
            num_rules = len(individual) // self.rule_len
            for r in range(num_rules):
                # get the rule
                rule = individual[r * self.rule_len: (r + 1) * self.rule_len]
                # classify the example
                res = self.rule_classify(rule, example)
                # return the result
                if res is not None: return res
            return None

    def test_accuracy(self, individual, data=None):
        '''Tests an individual on a dataset'''
        
        data = data or self.training
        corrects, incorrects = 0, 0
        # check if dataset is Iris
        if self.iris:
            for example in data:
                # get the class of the example
                class_example = example[len(self.inputs):]
                # encode the class of the example
                class_example = self.encode_data(class_example, True)
                # get the class of the individual
                class_individual = self.classify(individual, example)
                # convert bin to float
                if class_individual == class_example:
                    corrects += 1
                else:
                    incorrects += 1
        else:
            for example in data:
                # get the class of the example
                class_example = example[self.ante_len:]
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
        # get length of individual
        length = len(individual)
        # get the fitness
        fitness = accuracy ** 2 / length ** .5
        # fitness = accuracy ** 2
        # return the fitness
        return fitness

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

        # if self.debug:
        #     print('Fitnesses of Population: ', self.fitnesses)
        print('Best Individual: ', self.best)

        return self.best
    
    def generate_crossover_pts(self, parent, d1=None, d2=None):
        '''Generates crossover points
            CAN BE IMPROVED FOR EFFICIENCY
        '''
        # get the length of the parent
        upper_bound = len(parent) - 1
        # get the crossover points
        if d1 is None or d2 is None:
            # get candidate crossover points
            cpt1 = randint(0, upper_bound)
            cpt2 = randint(0, upper_bound)
            while cpt2 == cpt1: cpt2 = randint(0, upper_bound)
            # leftmost and rightmost crossover points
            leftmost = min(cpt1, cpt2)
            rightmost = max(cpt1, cpt2)
            # get distances d1 and d2
            d1 = leftmost % self.rule_len
            d2 = rightmost % self.rule_len
            # return the crossover points and distances
            return leftmost, rightmost, d1, d2
        # get the crossover points matching d1 and d2
        else:
            # get candidate crossover points
            cpt3_candidates = [i for i in range(0, len(parent)) 
                if i % self.rule_len == d1]
            if len(cpt3_candidates) == 0:
                raise Exception('Invalid crossover')
            cpt3 = choice(cpt3_candidates)
            cpt4_candidates = [i for i in range(0, len(parent))
                if i % self.rule_len == d2 and i != cpt3]
            if len(cpt4_candidates) == 0:
                raise Exception('Invalid crossover')
            # get first random crossover points in parents
            cpt4 = choice(cpt4_candidates)
            # get leftmost and rightmost crossover points
            leftmost = min(cpt3, cpt4)
            rightmost = max(cpt3, cpt4)
            # return the crossover points and distances
            return leftmost, rightmost, d1, d2

    # became expensive for iris
    def is_valid(self, individual):
        '''Checks if an individual is valid'''

        # get the length of the individual
        length = len(individual)
        # get the number of rules
        num_rules = length // self.rule_len
        # check if the individual is valid
        if length % self.rule_len != 0:
            return False
        if num_rules < 1 or num_rules > self.ruleset_len:
            return False
        if self.iris:
            # check each rule
            for r in range(num_rules):
                # get the rule
                rule = individual[r * self.rule_len: (r + 1) * self.rule_len]
                # check if the rule is valid
                if rule[self.ante_len:] == '11':
                    return False

                ante = rule[:self.ante_len]
                for i in range(0, self.ante_len, self.bin_len * 2):
                    # decompose antecedent rule
                    r_lower = ante[i:i+self.bin_len]
                    r_lower = self.bin_to_float_iris(r_lower)
                    r_upper = ante[i+self.bin_len:i+self.bin_len*2]
                    r_upper = self.bin_to_float_iris(r_upper)
                    # check if antecedent rule is valid
                    if r_lower > r_upper:
                        return False
            
        return True

    def crossover_op(self, parent1, parent2):
        '''Crossover between two parents rules to generate
           two children of variable length'''
       
        while True:
            try: # try to generate two valid children
                # get the crossover points of the parents
                cpt1, cpt2, d1, d2 = self.generate_crossover_pts(parent1)
                cpt3, cpt4, _, _ = self.generate_crossover_pts(parent2, d1, d2)

                # get the children (check if the crossover points are valid)
                child1 = parent1[:cpt1] + parent2[cpt3:cpt4] + parent1[cpt2:]
                child2 = parent2[:cpt3] + parent1[cpt1:cpt2] + parent2[cpt4:]
                # child1 = parent1[:cpt1] + parent2[cpt3:cpt4] + parent1[cpt1:cpt2] + parent2[cpt2:]
                # child2 = parent2[:cpt3] + parent1[cpt1:cpt2] + parent2[cpt3:cpt4] + parent1[cpt2:]

                # check if the children are valid
                if self.is_valid(child1) and self.is_valid(child2):
                    return child1, child2
                else:
                    raise Exception('Invalid crossover')

            except Exception as e:
                if e == 'Invalid crossover':
                    continue
        
    def get_probs(self):
        '''Returns the probabilities of each individual'''
        if len(self.fitnesses) == 0:
            self.evaluate()

        if len(self.probs) == 0:
            # get the sum of fitnesses
            sum_fitnesses = sum(self.fitnesses)
            if sum_fitnesses == 0:
                probs = [0.0 for _ in self.fitnesses]
                # raise Exception('Invalid sum of fitnesses')
            else:
                # get the probabilities
                probs = [f / sum_fitnesses for f in self.fitnesses]
            self.probs = probs

            return probs
        else:
            return self.probs

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
                self.population[pairs[i + 1]])
            children += twins

        # add the children to the population
        return children

    # TODO: implement
    def tournament_selection(self, population):
        '''Tournament selection of survivors'''
        pass

    # TODO: implement
    def rank_selection(self, population):
        '''Rank selection of survivors'''
        pass

    def proportional_selection(self, population):
        '''Fitness proportional selection of parents'''
       # get the population size
        size = len(population)
        # get the number of survivors to select
        k_survivors = round((1 - self.replace_rate) * size)
        # get probabilities
        probs = self.get_probs()
        # get the survivors 
        survivors = choices(population, probs, k=k_survivors)
        
        return survivors

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
            if len(new_population) != len(self.population):
                raise ValueError('Population size mismatch')
            else: self.population = new_population
            # evaluate
            self.evaluate()
            # keep the best individual
            # self.keep_best_individual()
            # stop early if threshold is reached
            if self.best[2] >= self.threshold:
                print('Threshold Reached')
                break
    
        # print the best individual
        print('Final Best Individual: ', self.best)
        self.print_individial(self.best[1])
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
            # if self.debug:
            #     print(f'here {attr}: {value}')
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
            index = int(value, 2)
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
        for r in range(0, len(individual), self.rule_len):
            rule = individual[r : r + self.rule_len]
            # check if it's iris
            if self.iris:
                # decode iris rule 
                rule = self.decode_rule_iris(rule)
            else:
                # decode the rule
                rule = self.decode_rule(rule)
            # print the rule
            print(rule, end='')

    def test(self, data):
        '''Test the algorithm by testing best individual'''
        # get the best individual
        if self.best is None: self.run()
        # get accuracy of the best individual
        accuracy = self.test_accuracy(self.best[1], data)
        # print the accuracy
        print(f'Accuracy: {accuracy * 100}%')
            
    def keep_best_individual(self):
        '''put the best back they were removed'''
        # check if best is present in population
        found = False
        for individual in self.population:
            if self.best is not None and \
                self.best[1] == individual:
                found = True
                break
        # if not found, add it
        if not found:
            # look for the worst individual based on
            # the fitness
            index_worst = self.fitnesses.index(min(self.fitnesses))
            # swap the best with the worst
            self.population[index_worst] = self.best[1]
            # self.population[0] = self.best[1]
            # maybe should swap with worse??
    
    def decode_rule_iris(self, rule):
        '''decode a rule for iris dataset'''
        res = ''
        # go through inputs
        for i in range(len(self.inputs)):
            # get the attribute
            attr = self.inputs[i]
            # get the lower and upper bounds values
            values = rule[i * self.bin_len * 2: (i + 1) * self.bin_len * 2]
            # get the lower bound
            lower = self.bin_to_float_iris(values[:self.bin_len])
            # get the upper bound
            upper = self.bin_to_float_iris(values[self.bin_len:])
            # adding to result
            res += f'({lower} <= {attr} <= {upper}) ^ '
        
        # remove the last '^ '
        res = res[:-3]
        res += ' => '
        # get the output value
        for i in range(len(self.outputs)):
            # get the attribute
            attr = self.outputs[i]
            # get the value
            value = rule[self.ante_len:]
            # convert to decimal
            index = int(value, 2)
            # get the attribute value
            res += f'{attr} = ({self.attributes[attr][index]}) ^ '
        # remove the last '^ '
        res = res[:-3]
        res += '\n'

        return res

    def is_match(self, ante_r: str, ante_e: list)-> bool:
        '''classify a rule for iris dataset'''
        for i in range(0, self.ante_len, self.bin_len * 2):
            # decompose antecedent rule
            r_lower = ante_r[i:i+self.bin_len]
            r_lower = self.bin_to_float_iris(r_lower)
            r_upper = ante_r[i+self.bin_len:i+self.bin_len*2]
            r_upper = self.bin_to_float_iris(r_upper)
            # get example value
            index = i // (self.bin_len * 2)
            e_value = float(ante_e[index])
            # check if it's a match
            if r_lower == 0.0 or r_upper == 0.0 or r_lower > r_upper or \
                not (r_lower <= e_value <= r_upper):
                return False
        return True
            

    def bin_to_float_iris(self, bin_str: str) -> float:
        '''convert a binary string to float'''
        # decompose the string
        int_part = bin_str[:self.int_len]
        dec_part = bin_str[self.int_len:]
        # get integer part of the number
        integer = int(int_part, 2)
        # get the decimal part of the number
        decimal = 0.0
        for pos in range(self.dec_len):
            bit = int(dec_part[pos])
            decimal += bit * 2 ** -(pos + 1)

        res = integer + decimal
        return res 