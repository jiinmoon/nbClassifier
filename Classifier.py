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

    # when reading, classifiy labels
    class ClassAttr(object):
        frequencyDict = {}
        totalDocsInClass = 0

        def __init__(self, label):
            pass

    _classLabelMap = []
    _classAttrs = {}
    _totalTrainDocs = 0
    _sizeOfVocabulary = 0

    _predictions = []
    _testLabels = []

    @staticmethod
    def new(arg):
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
        #print(f'{testData}')
        classAttrObj = self._classAttrs[classValue]

        frequencyDict = classAttrObj.frequencyDict
        totalDocsInClass = classAttrObj.totalDocsInClass
        result = totalDocsInClass/self._totalTrainDocs
        for word in testData:
            result *= ((frequencyDict.get(word, 0) + 1) / (sum(frequencyDict.values()) + self._sizeOfVocabulary))

        #print(f'P(c|d) = {result}')
        return result


    def setTrainData(self, trainData="", trainLabel=""):
        """ preprocessing on the train data.

        Opens the train data files and parses to componenets that are required
        for further computations down the line. Then, class attribute objects
        are created per each label and populates the necessary terms.

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
        assert len(self._predictions) > 0
        rawTestLabelDump = self._read_file(testLabels)
        formattedTestLabels = [line for line in rawTestLabelDump.split('\n')]
        corrects = [1 for x in zip(self._predictions, formattedTestLabels) if x[0] == x[1]]
        return (len(corrects) / len(self._predictions)) * 100

    # Need to be fixed
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
    trainData = os.getcwd() + '/traindata.txt'
    trainLabels = os.getcwd() + '/trainlabels.txt'
    testData = os.getcwd() + '/traindata.txt'
    testLabels = os.getcwd() + '/trainlabels.txt'

    #trainData = os.getcwd() + '/toyData.txt'
    #trainLabels = os.getcwd() + '/toyLabel.txt'
    #testData = os.getcwd() +'/toyTestData.txt'
    #testLabels = os.getcwd() + '/toyTestLabel.txt'

    #print(trainData, trainLabels)
    myClassifier = NBClassifier.new(NBClassifier.MODE_BERNOULI)
    myClassifier.setTrainData(trainData, trainLabels)
    #print(myClassifier)

    #singleTestData = ['Chinese', 'Chinese', 'Chinese', 'Tokyo', 'Japan']
    #prediction = myClassifier.predict(singleTestData)
    #print(f'{singleTestData} >>> {prediction}')
    predictions = myClassifier.predictSet(testData)
    accuracy = myClassifier.reportAccuracy(testLabels)

    print(accuracy)

if __name__ == '__main__':
    main()
