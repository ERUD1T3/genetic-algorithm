############################################################
#   Dev: Josias Moukpe
#   Class: Machine Learning
#   Date: 2/23/2022
#   file: testIrisReplacement.py
#   Description: vary replacement rate r 
#   [.1 to .9, .1 increment], output replacement 
#   rate andtest set accuracy for each of the 
#   three selection strategies.
#############################################################

# imports
from genetic import Genetic

def main():
    '''main of the program'''

    training_path = 'data/iris/iris-train.txt'
    testing_path = 'data/iris/iris-test.txt'
    attributes_path = 'data/iris/iris-attr.txt'
    population_size = 200 
    mutation_rate = .001
    # replacement_rate = .6 # from GABIL
    generations = 100
    fitness_threshold = 1.0 # used for early stopping
    selection_strategies = ['P', 'R', 'T'] # 'P' for proportional selection, 
                            # 'R' for rank selection,
                            # 'T' for tournament selection
    debug = False

    accuracies = []
    
    # vary the selection strategy
    for selection_strategy in selection_strategies:
        # print the selection strategy
        print('\nSelection Strategy:', selection_strategy)
        # vary the replacement rate
        for replacement_rate in range(.1, .9, .1):
            # print the replacement rate
            print('\nReplacement rate:', replacement_rate)
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
            accuracy = ga.test(ga.testing)

            accuracies.append(accuracy)
            
        print('\nAccuracies:', accuracies)
        accuracies.clear()
    
if __name__ == '__main__':
    main()
