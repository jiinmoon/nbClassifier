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
    myClassifier.predictSet(pathToTestData)
    myClassifier.reportAccuracy(pathToTestLabel)
    ...

"""

""" TODO:

[x] Finish file handling
[x] Pre-data processing
[x] Implement Multinomial
[] Implement Bernouli
[] Argparse
[] Switch input formats
[] PyDocs
[] Travis

"""

""" Notes:

- Multinomial implemented by default
- Override the computation functions in each respective classses.
    - Bernouli needs to be implemented

"""
import numpy as np

import os


class NBClassifier(object):
    """ NBClassifier is the Naive Bayes' Classifier for Text Classification purposes. """
    # Operating Mode :: Currently operates on MULTINOMIAL
    MODE_BERNOULI = 0
    MODE_MULTINOMIAL = 1

    class ClassAttr(object):
        """ individual class attributes retain own frequency dicts and sizes. """
        frequencyDict = {}
        totalDocsInClass = 0

        def __init__(self, label):
            pass

    _classLabelMap = [] # internal class values to index mapping.
    _classAttrs = {} # list of all class attributes
    _totalTrainDocs = 0 # total number of training documents
    _sizeOfVocabulary = 0 # total number of words within training set
    _predictions = []
    _testLabels = []

    @staticmethod
    def new(arg):
        """ factory method for initialization of NBClassifier. """
        if arg == 0:
            return Bernouli()
        elif arg == 1:
            return Multinomial()
        else:
            raise AssertionError("Cannot create classifier with given arg: {}".format(arg))

    def _read_file(self, filePath):
        """ opens file and read the file at given path.

        Args:
            filePath(str): absolute path to the file.

        Returns:
            str: data dump of given file (with both ends stripped)

        """
        with open(filePath) as f:
            fileContent = f.read()
        f.close()
        return fileContent.strip()

    def _count_word_frequency(self, data):
        """ counts the frequency of words within the data.

        Args:
            data(list([str])): list of documents which is comprised of ['word1', 'word2', 'word3', ... ]

        Returns:
            dict: dictionary where {key = 'word', value = number of occurrences}

        """
        _dict = {}
        for _docs in data:
            for _word in _docs:
                if _word in _dict:
                    _dict[_word] += 1
                else:
                    _dict[_word] = 1
        return _dict

    def _computeCondProb(self, testData, classValue):
        """ computes conditional probabilty based on given test data.

        It is implemented to use Multinomial formulae by default (needs refactor). The computation skips the
        alpha calculation.

        Args:
            testData(list(str)): test data set in the format of ['word1', 'word2', 'word3', ... ]
            classValue(int): indicator of class label

        Returns:
            float: P(T|c): the probability of given test data belong under classValue (w/o alpha)

        """
        classAttrObj = self._classAttrs[classValue]
        frequencyDict = classAttrObj.frequencyDict
        totalDocsInClass = classAttrObj.totalDocsInClass

        result = (totalDocsInClass/self._totalTrainDocs) # P(c)
        # Compute P(t|c) for each t in d
        for word in testData:
            result *= ((frequencyDict.get(word, 0) + 1) / (sum(frequencyDict.values()) + self._sizeOfVocabulary))
        return result


    def setTrainData(self, trainData="", trainLabel=""):
        """ preprocessing on the train data.

        Opens the train data files and parses to componenets that are required
        for further computations down the line. Then, class attribute objects
        are created per each label and populates the necessary terms.

        TODO:
            Should implement assert or fail indicator for setting train data.

        Args:
            trainData(str): absolute path to train data file
            trainLabel(str): absolute path to train label file

        """
        # 1. Read train data set from given file paths
        trainDataDump = self._read_file(trainData)
        trainLabelDump = self._read_file(trainLabel)

        # 2. Format train data and labels into list of words
        trainFormattedData = np.array([line.split(' ') for line in trainDataDump.split('\n')])
        trainFormattedLabels = np.array([label for label in trainLabelDump.split('\n')])

        # 3. Compute total unique words over all docs
        uniqueWords = set()
        for line in trainFormattedData:
            for w in line:
                uniqueWords.add(w)
        self._sizeOfVocabulary = len(uniqueWords)

        # 4. Instantiate classAttr object for each label
        for index, label in enumerate(set(trainFormattedLabels)):
            self._classLabelMap.append(label)
            # Distinguish only those that belong under the label
            trainDataInClass = trainFormattedData[[i == label for i in trainFormattedLabels]]
            # Size of total documents belong under the label
            totalTrainDocsInClass = len(trainDataInClass)
            # Update the total documents
            self._totalTrainDocs += totalTrainDocsInClass
            # Count the frequency (possibly replaced with Collections.Counter)
            frequencyDict = self._count_word_frequency(trainDataInClass)
            # Create new ClassAtr and set it.
            newClassAttrObj = NBClassifier.ClassAttr(label)
            self._classAttrs[index] = newClassAttrObj
            newClassAttrObj.frequencyDict = frequencyDict
            newClassAttrObj.totalDocsInClass = totalTrainDocsInClass

    def predict(self, testData=[]):
        """ predicts a singular case of test data.

        Args:
            testData(list(str)): list of words to classify the data.

        Returns:
            str: the most likely predicted label computed.

        """
        result = []
        for classValue in self._classAttrs:
            #print(f'Computing Label: {classValue}, {self._classLabelMap[classValue]}')
            result.append(self._computeCondProb(testData, classValue))
        return self._classLabelMap[result.index(max(result))]

    def predictSet(self, testData=""):
        """ predicts entire set of test data.

        Args:
            testData(str): absolute path to test data in text format.

        Returns:
            list(str): predictions of each given docs in ordered sequence.

        """
        rawTestDataDump = self._read_file(testData)
        formattedTestData = [line.split(' ') for line in rawTestDataDump.split('\n')]
        for test in formattedTestData:
            self._predictions.append(self.predict(test))
        return self._predictions

    def reportAccuracy(self, testLabels=""):
        """ computes accurracy of predictions against the given test labels.

        Args:
            testLabels(str): absolute path to test label file.

        Returns:
            float: percentage accuracy of prediction.

        """
        assert len(self._predictions) > 0
        rawTestLabelDump = self._read_file(testLabels)
        formattedTestLabels = [line for line in rawTestLabelDump.split('\n')]
        corrects = [1 for x in zip(self._predictions, formattedTestLabels) if x[0] == x[1]]
        return (len(corrects) / len(self._predictions)) * 100


class Bernouli(NBClassifier):
    """ Not used. """
    pass


class Multinomial(NBClassifier):
    """ Not used. """
    pass



def main():
    """ only use for quick testing purpose """

    trainData = os.getcwd() + '/data/traindata.txt'
    trainLabels = os.getcwd() + '/data/trainlabels.txt'

    #testData = os.getcwd() + '/data/traindata.txt'
    #testLabels = os.getcwd() + '/data/trainlabels.txt'

    testData = os.getcwd() + '/data/testdata.txt'
    testLabels = os.getcwd() + '/data/testlabels.txt'

    #trainData = os.getcwd() + '/data/toyData.txt'
    #trainLabels = os.getcwd() + '/data/toyLabel.txt'
    #testData = os.getcwd() +'/data/toyTestData.txt'
    #testLabels = os.getcwd() + '/data/toyTestLabel.txt'

    #print(trainData, trainLabels)
    myClassifier = NBClassifier.new(NBClassifier.MODE_BERNOULI)
    myClassifier.setTrainData(trainData, trainLabels)
    #print(myClassifier)

    #singleTestData = ['Chinese', 'Chinese', 'Chinese', 'Tokyo', 'Japan']
    #prediction = myClassifier.predict(singleTestData)
    #print(f'{singleTestData} >>> {prediction}')
    predictions = myClassifier.predictSet(testData)
    accuracy = myClassifier.reportAccuracy(testLabels)

    #print(predictions)
    print(accuracy)

if __name__ == '__main__':
    main()
