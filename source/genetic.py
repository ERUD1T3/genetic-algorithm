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
        self.training = training
        self.testing = testing
        self.attributes = attributes
        self.population = population
        self.mutation = mutation
        self.replacement = replacement
        self.max_generations = max_generations
        self.fitness_threshold = fitness_threshold
        self.selection_type = selection_type
        self.debug = debug

        # instanting params for the GA
        self.population = []
        self.fitness = []
        self.best_individual = None

    def __repr__(self) -> str:
        '''Returns a string representation of the GA'''
        pass

    def read_data(self, path):
        '''Reads data from a file'''
        pass

    def read_attributes(self, path):
        '''Reads attributes from a file'''
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