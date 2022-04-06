This is an implementation of Genetic algorithm with support for fitness-proportional, rank, 
and tournament selections, support for rule voting, iris dataset, and early stopping with 
fitness threshold.
    main.py                     (main file with options)
    genetic.py                  (dependency for genetic algorithm functionalities)
    utils.py                    (dependency for utility functions used)
    testTennis.py               (main tennis experiment file)
    testIris.py                 (main Iris experiment file)
    testIrisSelection.py        (main Iris with selection strategy experiment file)
    testIrisReplacement.py      (main Iris with replacement rate experiment file)
    data/                       (directory of all required attribute, training and testing files)

No need to compile since the Python files are interpreted

To run the tree with options, use the following command example:

$ python source/main.py \
-a data/tennis/tennis-attr.txt \
-d data/tennis/tennis-train.txt \
-t data/tennis/tennis-test.txt \
-p 500 \
-m .001 \
-r .6 \
-g 250 \
-f 1. \
-s P \
--debug

where python3 is the python 3.X.X interpreter, 
    optional arguments:
  -a ATTRIBUTES, --attributes ATTRIBUTES
                        path to the attributes files (required)
  -d TRAINING, --training TRAINING
                        path to the training data files (required)
  -t TESTING, --testing TESTING
                        path to the test data files (required)
  -p POPULATION, --population POPULATION
                        population size (required, must be even)
  -m MUTATION, --mutation MUTATION
                        mutation rate (required)
  -r REPLACEMENT, --replacement REPLACEMENT
                        replacement rate (required)
  -g GENERATIONS, --generations GENERATIONS
                        number of generations (required)
  -f FITNESS_THRESHOLD, --fitness-threshold FITNESS_THRESHOLD
                        fitness threshold (optional, If not provided, GA stops at the end of the generations)
  -s SELECTION, --selection SELECTION
                        selection strategy: P for fitness-proportional, T for tournament, R rank (default: 0)
  --debug               debug mode, prints statements activated (optional)

To find out about the options, use:
$ python3 main.py -h 

To run the different experiment files, use the following  command:

$ python3 testTennis.py > output.txt
$ python3 testIris.py > output.txt
$ python3 testIrisSelection.py > output.txt
$ python3 testIrisReplacement.py > output.txt

where python3 is the python 3.X.X interpreter, and provided the data files are present 
and in the same directory as the experiment files




