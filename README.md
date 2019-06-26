#  Simple Neural Network  #

A Python implementation of a 3-layer neural network from scratch, with multiple cost and activation
functions, using only the NumPy package. Evaluation on the MNIST dataset.


-- IMPORTANT --

The program was tested using Python-3.6.

The program reads the Train and Test Data from directories.
Since it only works with the MNIST dataset, it expects to find the Train and Test Dataset
in a specific location.

The training files, (i.e. train0.txt, train1.txt ...) should be in a separate directory without any other file.
Similarly, the test files (i.e., test0.txt, test1.txt ...) should be in another directory without any other files.

These 2 directories should be passed to the command line using the corresponding switches.

For example, one can provide the train data in a folder named 'Train' and the test data in a folder name 'Test'.
The path to these folders should be provided in the command line.

You can try running "C:\Program Files (x86)\Python36-32\python" Source/main.py --help for more info.

Example:

"C:\Program Files (x86)\Python36-32\python" Source/main.py  --normalize 1 --batch-size 20 --activation 1 --hidden-count 100 --learning-rate 0.1 --lambda-normalization 0.001 --grad-check 0 --epochs 10 --test-dir "Resources\Test" --train-dir "Resources\Train" --feature-count -1 --test-count -1