#!/usr/bin/python
""" unit test for Classifier.py


Author: fmoon

"""

from Classifier import NBClassifier

import unittest

class Test(unittest.TestCase):

    def test_construct_classifier(self):
        """ does classifiers return expected objects? """
        myClassifier = NBClassifier.new(NBClassifier.MODE_BERNOULI)
        assert not myClassifier == None
        myClassifier.saySomething()

Test().test_construct_classifier()
