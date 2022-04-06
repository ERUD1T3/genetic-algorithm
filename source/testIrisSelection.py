############################################################
#   Dev: Josias Moukpe
#   Class: Machine Learning
#   Date: 4/6/2022
#   file: testIrisSelection.py
#   Description: vary generation number, output 
#   generation number and test set accuracy for each
#   of the three selection strategies
#############################################################

# imports
from genetic import Genetic



def main():
    '''main of the program'''

    training_path = 'data/tennis/tennis-train.txt'
    testing_path = 'data/tennis/tennis-test.txt'
    attributes_path = 'data/tennis/tennis-attr.txt'
    population_size = 500 
    mutation_rate = .001
    replacement_rate = .6 # from GABIL
    # generations = 300
    fitness_threshold = 1.0 # used for early stopping
    selection_strategies = ['P', 'R', 'T'] # 'P' for proportional selection, 
                            # 'R' for rank selection,
                            # 'T' for tournament selection
    debug = False
    
    # vary the selection strategy
    for selection_strategy in selection_strategies:
        # print the selection strategy
        print('\nSelection Strategy:', selection_strategy)
        # vary the replacement rate
        for generations in range(100, 300, 50):
            # print the generations number
            print('\nGenerations:', generations)
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
            # print('\nTesting the best solution on training set...')
            # ga.test(ga.training)

            print('\nTesting the best solution on test set...')
            ga.test(ga.testing)

    

    
if __name__ == '__main__':
    main()
