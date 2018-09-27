#!/usr/bin/python3
""" Classifier.py


Classifier is a class that encapsulates the machine learning classification strategies by returning
a model object that learn from train set to predict the outcomes of test set. It may also report the
accuracy of its predictions.

As of now, Classifier only supports Naive Baye's Classification - in particular, text classification.


Usage:

    from Classifier import NBClassifier

    myClassifier = NBClassifier.new(NBClassifier.MODE_BERNOULI)
    myClassifier.setTrainData(pathToTrainData, pathToTrainLabel)
    ...


Author: fmoon

"""

""" TODO:

[] Finish file handling
[] Pre-data processing
[] Design scope of methods - which should be open? closed?
[] Internal structure - how abstract? treat as all attributes then decide on class?
[] A model (Classifier) is expected to:
    - train on given data set
    - predict on given data set
    - compute accuracy of previous computation
"""

""" Notes:

Refactor later on # of values that class attribute can hold: for now, it is binary.
The assignment input format is the de facto format to accept. Single CSV file should be easier to work
with but w/e, this is fine as well. For now, implement as if we have a seperate label text.

Per each class value, we need the dictionary.
Identify the what variables we require. TotalDocNum, ClassAttr, and so on.


"""
import os



class NBClassifier(object):
    MODE_BERNOULI = 0
    MODE_MULTINOMIAL = 1

    @staticmethod
    def new(arg):
        """ Factory method for instantiating the NBClassifier. """
        if arg == 0:
            return Bernouli()
        elif arg == 1:
            return Multinomial()
        else:
            raise AssertionError("Cannot create classifier with given arg: {}".format(arg))

    def _read_file(self, filePath):
        with open(filePath) as f:
            fileContent = f.read()
        f.close()
        return fileContent


    def setTrainData(self, trainData="", trainLabel=""):
        print('But this is inherited none-the-less')



class Bernouli(NBClassifier):





class Multinomial(NBClassifier):




def main():
    """ small testing purposes only """

    # print(os.getcwd())
    trainData = os.getcwd() + '/traindata.txt'
    trainLabels = os.getcwd() + '/trainlabels.txt'

    print(trainData, trainLabels)
    myClassifier = NBClassifier.new(MODE_BERNOULI)


if __name__ == '__main__':
    main()
