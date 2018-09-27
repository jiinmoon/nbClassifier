#!/usr/bin/python3
"""NBMultinomial

    NBMultinomi is a python script that performs the Naive Bayes' Text Classification on given set of data. It is created for partial fullfillment of assignment #1 of the SENG474: Data Mining course.

    Try to be as generic as possible to work on most cases.

    Specified data formats to be accepted?
    Sys.path corrections.
    full commetings.
    TODO!!

    Usages:

        ./nbClassifier -traindata file1.txt -trainlabels file2.txt -testdata file3.txt -testlabels file4.txt

Author: fmoon

"""

"""TODO:
    * Divide between main.py and Classifier object.
    * The instance of Classifier provides the capabilities of the classification.
    * Instantiate. Set train/label. Generate model.
    * Once ready. Perform the
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

    def saySomething(self):
        print('I am Bernouli!')

class Multinomial(NBClassifier):

    def saySomething(self):
        print('I am Multinomial!')



def main():
    """ used for testing purposes only for now """
    print('NBClassifier for SENG 474')

    # print(os.getcwd())
    trainData = os.getcwd() + '/traindata.txt'
    trainLabels = os.getcwd() + '/trainlabels.txt'

    print(trainData, trainLabels)
    myClassifier = NBClassifier.create(BERNOULI)
    myClassifier.saySomething()
    pass



if __name__ == '__main__':
    main()
