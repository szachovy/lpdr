#!/usr/bin/env python
# -*- coding: utf-8 -*-

__version__ = '0.1'
__author__ = 'Wiktor Maj'

#------
import unittest
import subprocess
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from lpdr import lpdr
#------
 

def process(image):
    '''
        Execute command-line request and gather standard streams from them.
        
        Args:
            image: graphical file as an input to lpdr program.
        
        Returns:
            stdout: standard output from the program execution
            stderr: standard error from the program execution
    '''
    process = subprocess.Popen(['python', 'lpdr/lpdr.py', '-i', '{}'.format(image)],
                         stdout=subprocess.PIPE, 
                         stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return stdout, stderr
 
class LPD(unittest.TestCase):
    '''
        Test case class for checking input and output matrices of LPD. 
    '''
    def setUp(self):
        ''' LPD detection init'''
        self.detector = lpdr.LPD()
        self.detector.detect()
        
    def test_check_in(self):
        '''
            Check if the input matrix match specified channels for prediction
            with a specified range of values.
        '''
        self.assertEqual(self.detector.X.shape, (1, self.detector.width, self.detector.height, 3))
        self.assertTrue(np.all(self.detector.X >= 0))
        self.assertTrue(np.all(self.detector.X <= 1))

    def test_check_out(self):
        '''
           Check if the input matrix match specified channels for license plate extraction
           with a specified range of values. 
        '''
        self.assertEqual(self.detector.Y.shape, (self.detector.size[1], self.detector.size[0], 3))
        self.assertTrue(np.all(self.detector.Y >= 0))
        self.assertTrue(np.all(self.detector.Y <= 1))

class LPR(unittest.TestCase):
    '''
        Test case class for checking input matrices and output strings of LPR. 
    '''
    def setUp(self):
        ''' LPR recognizer init '''
        self.recognizer = lpdr.LPR()

    def test_check_in(self):
        '''
            Check if the input matrix (processed output matrix from LPD) 
            match specified shapes and values to propagate the OCR.
        '''
        self.assertEqual(self.recognizer.Y.shape, (self.recognizer.size[1], self.recognizer.size[0], 3))
        self.assertTrue(np.all(self.recognizer.Y >= 0))
        self.assertTrue(np.all(self.recognizer.Y <= 255))
        
    def test_check_out(self):
        '''
            Checking if output strings are appropriate to be processed as a license plate
            characters to be extracted as program output.
        '''
        self.assertTrue(self.recognizer.output.isalnum())
        self.assertIsInstance(self.recognizer.output, str)
        self.assertTrue(self.recognizer.output.isupper())
        self.assertIn(len(self.recognizer.output), range(4, 11)) 



def test_on_sample(image, label):
    '''
        Test if the given picture output 
        match the label output defines by a human. 
        
        Args:
            image: name of the image in the test_images folder.
            label: string verified by human as a license plate numbers.
            
        Returns:
            Test_On_Sample unittest case.
    '''   
    class Test_On_Sample(unittest.TestCase):     
        def test(self):
            '''
                Compares the program stdout with label
                
                Raises:
                    Exception if the human specified output does not match
                    the output recognized by the program. 
            '''
            stdout, stderr = process('test_images/{}'.format(image))
            stdout = str(stdout).split(u'\u0027')[1][:-2]
            try:
                self.assertEqual(stdout, label)
            except AssertionError:
                print('The desired output should be {}, but {} was returned from {}'.format(label, stdout, image))
                print('---- Log: ----')
                print(stderr)
                raise
                
    return Test_On_Sample

# to make test_on_sample on a specific image and runs LPD and LPR on it, 
# name it image.png or other default as like in the lpdr.py, and place it to the direct execution catalog.
class Test_1(test_on_sample('image1.png', 'GD4500H')): pass
class Test_2(test_on_sample('image2.jpg', 'DBI47861')): pass
class Test_3(test_on_sample('image3.jpg', 'KNSVWO4')): pass
class Test_4(test_on_sample('image4.jpg', 'PO2KS60')): pass
class Test_5(test_on_sample('image5.jpg', 'MPE3389')): pass

    
if __name__ == '__main__':
    unittest.main()
        
