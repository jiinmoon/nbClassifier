#!/usr/bin/python
""" unit test for Classifier.py """

from Classifier import NBClassifier

import os
import unittest

class Test(unittest.TestCase):

    def test_construct_classifier(self):
        """ does classifiers return expected objects? """

        myClassifier_Bernouli = NBClassifier.new(NBClassifier.MODE_BERNOULI)
        myClassifier_Multinomial = NBClassifier.new(NBClassifier.MODE_MULTINOMIAL)
        assert not myClassifier_Bernouli == None and not myClassifier_Multinomial == None
        assert 'Bernouli' in myClassifier_Bernouli.__repr__() and 'Multinomial' in myClassifier_Multinomial.__repr__()


    def test_toy_example(self):
        """ test classifier on toy example """

        trainData = os.getcwd() + '/data/toyData.txt'
        trainLabels = os.getcwd() + '/data/toyLabel.txt'

        myClassifier = NBClassifier.new(NBClassifier.MODE_MULTINOMIAL)
        myClassifier.setTrainData(trainData, trainLabels)

        # Predict singluar case
        singleTestData = ['Chinese', 'Chinese', 'Chinese', 'Tokyo', 'Japan']
        self.assertTrue('1' == myClassifier.predict(singleTestData))

    def test_train_example(self):
        """ test on real set. """
        pass

# Increasing verbosity
suite = unittest.TestLoader().loadTestsFromTestCase(Test)
unittest.TextTestRunner(verbosity=2).run(suite)
