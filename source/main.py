############################################################
#   Dev: Josias Moukpe
#   Class: Machine Learning
#   Date: 2/23/2022
#   file: main.py
#   Description: main file to run the program
#############################################################

# imports
import argparse
from genetic import Genetic

def parse_args():
    '''parse the arguments for genetic algorithm'''

    parser = argparse.ArgumentParser(
        description='Genetic Algorithm for Machine Learning'
    )

    parser.add_argument(
        '-a', '--attributes',
        type=str,
        required=True,
        help='path to the attributes files (required)'
    )

    parser.add_argument(
        '-d', '--training',
        type=str, 
        required=True,
        help='path to the training data files (required)'
    )
    
    parser.add_argument(
        '-t', '--testing',
        type=str , 
        required=True,
        help='path to the test data files (required)'
    )

    parser.add_argument(
        '-p', '--population',
        type=int,
        required=True,
        help='population size (required)'
    )

    parser.add_argument(
        '-m', '--mutation',
        type=float, 
        required=True,
        help='mutation rate (required)'
    )

    parser.add_argument(
        '-r', '--replacement',
        type=float, 
        required=True,
        help='replacement rate (required)'
    )

    parser.add_argument(
        '-g', '--generations',
        type=int, 
        required=True,
        help='number of generations (required)'
    )

    parser.add_argument(
        '-f', '--fitness-threshold',
        type=float, 
        required=False,
        help='fitness threshold (optional, \
            If not provided, GA stops at the end of the generations)',
    )

    parser.add_argument(
        '-s', '--selection',
        type=str, 
        required=False,
        default="P",
        help='selection strategy: P for fitness-proportional, \
             T for tournament, R rank (default: 0)',
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        default=False,
        help='debug mode, prints statements activated (optional)'
    )

    # parse arguments
    args = parser.parse_args()
    return args


def main():
    '''main of the program'''
    args = parse_args() # parse arguments
    print(' args entered',args)

    training_path = args.training
    testing_path = args.testing
    attributes_path = args.attributes
    population_size = args.population
    mutation_rate = args.mutation
    replacement_rate = args.replacement
    generations = args.generations
    fitness_threshold = args.fitness_threshold
    selection_strategy = args.selection
    debug = args.debug
    
    print('\nCreating a Genetic Algorithm object...')
    ga = Genetic(
        training_path,
        testing_path,
        attributes_path,
        population_size,
        mutation_rate,
        replacement_rate,
        generations,
        fitness_threshold,
        selection_strategy,
        debug
    )

    # print genetic algorithm object
    print('\nGenetic Algorithm object created:')
    print(ga)
    
    # run the genetic algorithm
    print('\nRunning the Genetic Algorithm...')
    ga.run()
    print('\nDone!')

    # test best solution
    print('\nTesting the best solution on training set...')
    ga.test(ga.training)

    print('\nTesting the best solution on test set...')
    ga.test(ga.testing)

    

    
if __name__ == '__main__':
    main()
