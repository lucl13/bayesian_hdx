'''
Test objects and functions in analysis.py
'''
from __future__ import print_function
import utils
import unittest
import os

input_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'input'))

print(input_dir)

class TestParseOutputFile(untitest.TestCase):
    '''
    ParseOutputFile reads an output file created from a sampler
    '''

    self.output_file = input_dir

    def test_create_pof(self):
        pass


class TestOutputAnalysis(untitest.TestCase):

    def test_concatenate_output_files(self):
        pass
