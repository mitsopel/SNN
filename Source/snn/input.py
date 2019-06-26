import snn.layer
import snn.math
import sys
import numpy as np

from os import listdir
from os.path import isfile, join

def GetFileLineCount(Filename):
    with open(Filename) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def shuffle_alike(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

def LoadDummyFeatures(Count, FeatureCount, HotIndex):
#    Features = np.random.uniform(low=0, high=1, size=(Count, FeatureCount))
    Features = np.ones((Count, FeatureCount))
    Labels = np.zeros((Count, 10))
    for Label in Labels:
        Label[HotIndex] = 1;

    return (Features, Labels)

def LoadFeaturesFromFile(Filename, HotIndex, Count, Normalize):

    normalize = np.vectorize(snn.math.normalize)  # or use a different name if you want to keep the original f

    if ( Count < 0  ):
        FeatureCount = GetFileLineCount(Filename)
    else:
        FeatureCount = Count

    ReadFeatures = []
    for LineIndex, Line in enumerate( open(Filename, "r") ):
        if (LineIndex < FeatureCount):
            ReadFeatures += Line.split()
        else:
            break;

    # Initialize Features and Labels
    Features = np.asfarray(ReadFeatures)
    Labels = np.zeros((FeatureCount, 10))

    # Set Hot Index
    for Label in Labels:
        Label[HotIndex] = 1

    # Normalize Values?
    if ( Normalize ):
        Max = np.amax(Features)
        Features = normalize(Features, Max)

    Features.shape = (FeatureCount, Features.size//FeatureCount)

#    print(Features.shape)

    return (Features, Labels)

def LoadFeaturesFromDirectory(Dirname, CountPerFile, Normalize):
    FilesInDir = [Filename for Filename in listdir(Dirname) if isfile(join(Dirname, Filename))]

    FeatureCount = 0

    FirstFile = FilesInDir[0]
    LastDotIndex = FirstFile.rfind(".")
    HotIndex = int(FirstFile[LastDotIndex-1])
    Features, Labels = LoadFeaturesFromFile(Dirname+'//'+FirstFile, HotIndex, CountPerFile, Normalize)

    for Filename in FilesInDir[1:]:
        LastDotIndex = Filename.rfind(".")
        HotIndex = int(Filename[LastDotIndex-1])

        NewFeatures, NewLabels = LoadFeaturesFromFile(Dirname+'//'+Filename, HotIndex, CountPerFile, Normalize)

        Features = np.concatenate((Features, NewFeatures), axis=0)
        Labels = np.concatenate((Labels, NewLabels), axis=0)

        sys.stdout.flush()

    # Possibly shuffle input
    # shuffle_alike(Features, Labels)

    return (Features.tolist(), Labels.tolist())
