#!/usr/bin/python
""" unit test for Classifier.py """

from Classifier import NBClassifier

import unittest

class Test(unittest.TestCase):

    def test_construct_classifier(self):
        """ does classifiers return expected objects? """

        # 1. Testing Bernouli & Multinomial Models
        myClassifier_Bernouli = NBClassifier.new(NBClassifier.MODE_BERNOULI)
        myClassifier_Multinomial = NBClassifier.new(NBClassifier.MODE_MULTINOMIAL)
        assert not myClassifier_Bernouli == None and not myClassifier_Multinomial == None
        assert 'Bernouli' in myClassifier_Bernouli.__repr__() and 'Multinomial' in myClassifier_Multinomial.__repr__()





if __name__ == '__main__':
    unittest.main()
