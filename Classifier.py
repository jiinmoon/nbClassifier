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

[x] Finish file handling
[] Pre-data processing
[] Design scope of methods - which should be open? closed?
[] Internal structure - how abstract? treat as all attributes then decide on class?
[] A model (Classifier) is expected to:
    - train on given data set
    - predict on given data set
    - compute accuracy of previous computation
[] Split the single model into Bernouli and Multinomial
    - Implement as Multinomial first.
    - Identify common methods and extract override methods specific to
    each classifier model.
[] Think about creating an attribute class that holds values.
    - especially for class attribute will help a lot.
    - if such, we will have to ask user for which attribute is the class attribute.
        - we won't since in this particular instance, we are getting it seperately as
        trainLabel.txt

"""

""" Notes:

Refactor later on # of values that class attribute can hold: for now, it is binary.
The assignment input format is the de facto format to accept. Single CSV file should be easier to work
with but w/e, this is fine as well. For now, implement as if we have a seperate label text.

Per each class value, we need the dictionary.
Identify the what variables we require. TotalDocNum, ClassAttr, and so on.


"""
import numpy as np

import os


# c can be any values in C. There should be a mapping :: such that internally, we work with integer mapped to c.
# But for now, assume the binary classification (c or not c). We will abstract it out later.
class NBClassifier(object):
    MODE_BERNOULI = 0
    MODE_MULTINOMIAL = 1

    _frequencyDict0 = {}
    _frequencyDict1 = {}
    _totalTrain0 = 0
    _totalTrain1 = 0
    _sizeVocabulary = 0
    _totalTrainDocs = 0

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
        return fileContent.strip()

    def _count_word_frequency(self, data):
        _dict = {}
        for _docs in data:
            for _word in _docs:
                if _word in _dict:
                    _dict[_word] += 1
                else:
                    _dict[_word] = 1
        return _dict

    def _computeCondProb(self, testData, classValue):
        """ Multinomial by default.

        skip alpha - irrelevant

        Suppose classValue = 1

        We need following terms:
            P(c)
            P(t|c) for each t in the testData.
            for t in testData:
                _probTgivenC *= ( (frequencyDictC1[t] + 1) / (sizeWords1 + sizeVocabulary) )

        also include P(c) for P(c|d) computation
        """

        print(f'{testData}')
        frequencyDict = self._frequencyDict1 if classValue else self._frequencyDict0
        totalTrainC = self._totalTrain1 if classValue else self._totalTrain0
        result = totalTrainC/self._totalTrainDocs
        for word in testData:
            print(f'Computing P({word}|c)')
            result *= ((frequencyDict.get(word, 0) + 1) / (sum(frequencyDict.values()) + self._sizeVocabulary))

        print(f'P(c|d) = {result}')
        return result


    def setTrainData(self, trainData="", trainLabel=""):
        """ preprocessing on the train data.

        Opens the train data files and parses to componenets that are required
        for further computations down the line.

        It will populate following attributes:
            totalNumberDocuments (int)
            frequencyDictionaries {t: frequency of t}
            sizeOfVocabulary (int)

        Args:
            trainData(str): absolute path to train data file
            trainLabel(str): absolute path to train label file

        Returns:
            bool: true if accepted; false otherwise.

        """

        # Identify the unique labels.
        # Create class_attr obj per each unique labels.
        # it will hold all data necessary for computation.

        trainDataDump = self._read_file(trainData)
        trainLabelDump = self._read_file(trainLabel)

        #print('Train Data: {}'.format(trainDataDump))
        #print('Train Label: {}'.format(trainLabelDump))

        #print([(x.split(' '), int(y)) for x, y in zip(trainDataDump.split('\n'), trainLabelDump.split('\n'))])
        #print([x.split(' ') for x in trainDataDump.split('\n')])

        trainData = np.array([x.split(' ') for x in trainDataDump.split('\n')])
        trainLabel = np.array([int(y) for y in trainLabelDump.split('\n')])
        #print(trainData)
        #print(trainLabel)
        #print(trainData[trainLabel == 1])
        #print(trainData[trainLabel == 0])
        #print(list(zip(list(trainDataDump.split('\n'), list(trainLabelDump.split('\n')))))

        trainData1 = trainData[trainLabel == 1]
        trainData0 = trainData[trainLabel == 0]
        #print(len(trainData1), len(trainData0))

        self._totalTrain0 = len(trainData0)
        self._totalTrain1 = len(trainData1)
        self._totalTrainDocs = len(trainData1) + len(trainData0)
        #print(self._totalTrainDocs)

        self._frequencyDict1 = self._count_word_frequency(trainData1)
        self._frequencyDict0 = self._count_word_frequency(trainData0)
        #print(self._frequencyDict1, self._frequencyDict0)

        sizeWords1 = sum(self._frequencyDict1.values())
        sizeWords0 = sum(self._frequencyDict0.values())
        self._sizeVocabulary = len(
                set(self._frequencyDict1).union(
                    set(self._frequencyDict0)))
        #print(sizeWords1, sizeWords0, self._sizeVocabulary)

    def predict(self, testData=[]):
        """ testData is a list of words.

        This is a prediction of singluar case, that is where does d classifies under?

        Abstractly, we compute P(c | d) for ea c in C (in this case only two cases: c or not c).
        Then, we choose max val amongst them and report corresponding c.

        P(c|d) = alpha * P(c) * productSum(P(t|c) for all t in d)

        """
        C = [1, 0]
        result = []
        for classValue in C:
            result.append(self._computeCondProb(testData, classValue))
        return max(result)

    def __str__(self):
        formatter = """ Train Data Information:
            \tfrequencyDict1: {}
            \tfreqeuncyDict0: {}
            \ttotalTrainDocs: {}
            \tsizeVocabulary: {}
            """

        return formatter.format(self._frequencyDict1, self._frequencyDict0,
                    self._totalTrainDocs, self._sizeVocabulary)


class Bernouli(NBClassifier):
    """ Not used. """
    pass



class Multinomial(NBClassifier):
    """ Not used. """
    pass



def main():
    """ small testing purposes only """

    # print(os.getcwd())
    #trainData = os.getcwd() + '/traindata.txt'
    #trainLabels = os.getcwd() + '/trainlabels.txt'

    trainData = os.getcwd() + '/toyData.txt'
    trainLabels = os.getcwd() + '/toyLabel.txt'

    print(trainData, trainLabels)
    myClassifier = NBClassifier.new(NBClassifier.MODE_BERNOULI)
    myClassifier.setTrainData(trainData, trainLabels)
    print(myClassifier)

    testData = ['Chinese', 'Chinese', 'Chinese', 'Tokyo', 'Japan']
    prediction = myClassifier.predict(testData)
    print(f'{testData} >>> {prediction}')


if __name__ == '__main__':
    main()
