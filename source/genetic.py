############################################################
#   Dev: Josias Moukpe
#   Class: Machine Learning
#   Date: 2/22/2022
#   file: genetic.py
#   Description: : main genetic algorithm file
#############################################################

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
        self.population = population
        self.mutation = mutation
        self.replacement = replacement
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

        # instanting params for the GA
        self.population = []
        self.fitness = []
        self.best_individual = None

    def __repr__(self) -> str:
        '''Returns a string representation of the GA'''
        res = 'Genetic Algorithm\n'
        res += '- Population: {}\n'.format(self.population)
        res += '- Mutation: {}\n'.format(self.mutation)
        res += '- Replacement: {}\n'.format(self.replacement)
        res += '- Max Generations: {}\n'.format(self.max_generations)
        res += '- Fitness Threshold: {}\n'.format(self.fitness_threshold)
        res += '- Selection Type: {}\n'.format(self.selection_type)
        
        # might be useful to print the induviduals of the population

        return res


    # TODO: test this function
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

     # TODO: test this function
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

    def encode(self, individual):
        '''Encodes an individual
            data instance -> binary encoded instance
        '''
        encoded = ""

        for attr in self.inputs:
            if individual[attr] == '1':
                encoded += '1'
            else:
                encoded += '0'


    def decode(self, individual):
        '''Decodes an individual'''
        pass

    def discretize(self, data):
        '''Discretizes the data'''
        pass

    def run(self):
        '''Run the Genetic Algorithm'''
        pass

    def crossover(self, parent1, parent2):
        '''Crossover between two parents'''
        pass

    def mutation(self, child):
        '''Mutation of a child'''
        pass


    def selection(self, population, fitness):
        '''Selection of parents based on the type 
        of selection chosen'''
        pass


    def fitness(self, individual):
        '''Measures Fitness of an individual'''
        pass


    def test(self, individual, data=None):
        '''Tests an individual on a dataset'''
        
        data = data or self.testing