# Import SNN related libraries
import snn.input
import snn.math
import snn.neuralnet

# System libraries
import argparse
import sys

# Mathematics
import numpy as np

# Parser State
SNNParser = argparse.ArgumentParser(description='Process some integers.')
SNNParser.add_argument('--epochs', '-ep',  type=int, default=10, nargs='?', help='Number of Epochs', metavar='ep')
SNNParser.add_argument('--batch-size', '-bs', type=int, default=100, nargs='?', help='Batch Size', metavar='bs')
SNNParser.add_argument('--hidden-count', '-hc', type=int, default=100, nargs='?', help='Number of nodes in the Hidden Layer', metavar='hc')
SNNParser.add_argument('--feature-count', '-fc', type=int, default=-1, nargs='?', help='Number of features to use for training, use -1 for all available', metavar='fc')
SNNParser.add_argument('--test-count', '-tc', type=int, default=-1, nargs='?', help='Number of features to use for testing, use -1 for all available', metavar='tc')
SNNParser.add_argument('--test-dir', '-tf', help='Directory with Test Files', metavar='tf')
SNNParser.add_argument('--train-dir', '-rf', help='Directory with Train Files', metavar='rf')
SNNParser.add_argument('--activation', '-ac', type=int, default=1, nargs='?', help='Hidden Layer activation function 1=sigmoid, 2=relu, 3=softplus, 4=cos, 5=tanh', metavar='ac')
SNNParser.add_argument('--learning-rate', '-lr', type=float, default=0.01, nargs='?', help='Hidden Layer activation function 1=sigmoid, 2=relu, 3=softplus, 4=cos, 5=tanh', metavar='lr')
SNNParser.add_argument('--lambda-normalization', '-ln', type=float, default=0.001, nargs='?', help='Lambda Normalization factor', metavar='ln')
SNNParser.add_argument('--grad-check', '-gc', type=int, default=0, help='Whether to perform Gradient Check')
SNNParser.add_argument('--normalize', '-nm', type=int, default=1, help='Normalize Data to [0,1]')
SNNParser.print_help()
SNNArgs = SNNParser.parse_args()

print(SNNArgs)

if ( SNNArgs.train_dir == None or SNNArgs.test_dir == None ):
    print()
    print("Please provide directory for train and test data")
    exit()

Activation = SNNArgs.activation
ActivationFunction = None

LambdaNormalization = SNNArgs.lambda_normalization
LearningRate = SNNArgs.learning_rate
Epochs = SNNArgs.epochs
BatchSize = SNNArgs.batch_size
RunGradCheck = bool(SNNArgs.grad_check)
HiddenLayerCount = SNNArgs.hidden_count
FeatureCount = SNNArgs.feature_count
TestCount = SNNArgs.test_count
NormalizeData = bool(SNNArgs.normalize)
TestDir = SNNArgs.test_dir
TrainDir = SNNArgs.train_dir

if ( RunGradCheck ):
    snn.math.grad_check_all()

# Select activation function
if ( Activation == 1 ):
    ActivationFunction = snn.math.sigmoid
elif ( Activation == 2 ):
    ActivationFunction = snn.math.relu
elif ( Activation == 3 ):
    ActivationFunction = snn.math.softplus
elif ( Activation == 4 ):
    ActivationFunction = snn.math.cos
elif ( Activation == 5 ):
    ActivationFunction = snn.math.tanh
else:
    ActivationFunction = snn.math.sigmoid

# Print program info
print ("Training with:")
print ("Epochs                   = {0}".format(Epochs))
print ("Learning Rate            = {0}".format(LearningRate))
print ("Lambda Normalization     = {0}".format(LambdaNormalization))
print ("Mini-Batch Size          = {0}".format(BatchSize))
print ("Hidden Layer Node Count  = {0}".format(HiddenLayerCount))
print ("Test Count               = {0}".format(TestCount))
print ("Feature Count            = {0}".format(FeatureCount))
print ("Test Dir                 = {0}".format(TestDir))
print ("Train Dir                = {0}".format(TrainDir))
print ("Run Gradient Check       = {0}".format(RunGradCheck))
print ("Normalize Data           = {0}".format(NormalizeData))
print ("Hidden Activation        = {0}".format(ActivationFunction.__name__))

# For debugging purposes initially seed numpy.random with a Constant Value
np.random.seed(1)

# Load Data
TrainData = snn.input.LoadFeaturesFromDirectory(TrainDir, FeatureCount, NormalizeData)
TestData = snn.input.LoadFeaturesFromDirectory(TestDir, TestCount, NormalizeData)

PackedTrainData = [(TrainData[0][i], TrainData[1][i]) for i in np.arange(len(TrainData[0]))]
PackedTestData = [(TestData[0][i], TestData[1][i]) for i in np.arange(len(TestData[0]))]

# Create 3-layer neural network
SNN = snn.neuralnet.NeuralNet( [784,HiddenLayerCount,10], Activation = ActivationFunction )

# Train network on MNIST dataset
SNN.train( PackedTrainData, Epochs, BatchSize, LearningRate, LambdaNormalization, TestData = PackedTestData )
