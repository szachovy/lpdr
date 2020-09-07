#!/usr/bin/env python
# -*- coding: utf-8 -*-

__version__ = '0.1'
__author__ = 'Wiktor Maj'

#------
import numpy as np
import cv2

from keras.models import model_from_json
from pytesseract import image_to_string

import argparse
import os
import stat
import pkg_resources
#------

### tesseract utility for Windows
#pytesseract.pytesseract.tesseract_cmd = 'absolute_path_to_tesseract.exe'

### defaults for argparser (uncomment and change if needed)
#DEFAULT_CAPTURE=False
#DEFAULT_IMAGE_PATH='image.png' 
DEFAULT_NET_PATH='lpdr/wpod_net/wpod_net_update1' # see self.__net_path

def readable(path):
    ''' 
        Check if the source path for the captured image or wpod_net is readable.
        
        Args:
            path: relative path to the destination file.

        Returns:
            Boolean flag whatever the path is readable or not.
    '''
    st = os.stat(path)
    return bool(st.st_mode & stat.S_IRGRP)
    
class ArgParser(type):
    '''
        Parsing utility, which sends the arguments passed from cli to LPD as a class argument.
    '''
    def __new__(cls, name, parents, dct):
        '''
            Create an object with a capture flag and image_path directly.
            
            Args:
                name: is the name of the class to be created
                parents: is the list of the class's parent classes
                dct: is the list of class's attributes (methods, static variables)
            
            Returns:
                Return object as an instance of LPD class.
                
            Raises:
                Exception: If the capture argument not passed correctly
        '''
        parser = argparse.ArgumentParser()
        try:
            parser.add_argument('-c', '--capture', default=DEFAULT_CAPTURE, help="Captures image from camera after run")
        except NameError:        
            parser.add_argument('-c', '--capture', default=False, help="Captures image from camera after run")
        try:
            parser.add_argument('-i', '--image_path', default=DEFAULT_IMAGE_PATH, help="Defines input/output path for image")
        except NameError:
            try:
                readable('image.jpg')
                parser.add_argument('-i', '--image_path', default='image.jpg', help="Defines input/output path for image")
            except FileNotFoundError:
                parser.add_argument('-i', '--image_path', default='image.png', help="Defines input/output path for image")
            
        args = parser.parse_args()
        
        try:
            dct['capture'] = int(args.capture)
        except ValueError:
            try:
                dct['capture'] = True if args.capture in ['True', 'T', 'Y', 'Yes'] else False
            except: 
                raise Exception('capture argument not passed correctly')
        dct['image_path'] = args.image_path
   
        return super(ArgParser, cls).__new__(cls, name, parents, dct)
        
class SparseList(dict):
    '''
        Dict derivative created with the purpose of assurance that 
        key point labels are passed correctly before license plate extraction.
        Sparsity is defined as a number of results according to confidence
        during destination points passing.
    '''
    def __init__(self):
        self.labels = {}
            
    def __setitem__(self, idx, val):
        '''
            Every value point is set to the label`s collection 
            if only if all of them are not None and have the same
            shapes in a given destination.
            
            Args:
                idx: element index in labels
                val: value to insert into labels in given idx
            
            Raises:
                Exception: Conditions of a given label are not satisfied to be evaluated
        '''
        if (val['pts'].shape == (2, 4)) and (val['tl'].shape == (2, )) and (val['br'].shape == (2, )) and (val['prob'].shape == ()):
            self.labels[idx] = val
        else:
            raise Exception('matrix shapes does not match for the next stages, please change image settings')

    def __getitem__(self, idx):
        '''
            Args:
                idx: element index in labels
                
            Returns:
                Coordinate inserted into the collection.
                
            Raises:
                IndexError: If omitted coordinates are passed to the next step.
        '''
        try:
            return self.labels[idx]
        except KeyError:
            raise IndexError('uncomplete data, unable to evaluate')

    def get(self):
        '''
            Returns:
                Dict of labels filled by setitem from pred
        '''
        return self.labels
        
class LPD(metaclass=ArgParser):
    '''
    
    The first part of LPDR

    With given settings:    
        - Load the image from an existent source or take the photo of the car (by default) or motorcycle.
        - Process the image to the appropriate form.
        - Load the wpod_net model and predict given a matrix.
        - Find the points where the probability of an object is above a given threshold
        - Collect affines from the corresponding point and transform to coordinates
        - Non-Max-Suppression
        - Find T-matrix from transformed label points
        - Process perspective distance to output image

    '''
    def __init__(self):
        ##### webcam settings #####
        # camera source
        self.__source = 0
        
        # width of image to be captured (in px)
        self.__width = 480
        
        # height of image to be captured (in px)
        self.__height = 480
        
        ##### storage #####
        # input matrix
        self._X = None

        # output matrix
        self._Y = None

        # rectangular or squared size of the plate
        self._size = None
        
        ##### detection settings #####
        # destination path to wpod_net files without extensions
        try:
            self.__net_path = DEFAULT_NET_PATH
        except NameError:
            self.__net_path = None
        
        # iou threshold in nms operation
        self.__iou_threshold = 0.1 
        
        # degree of confidence that the detection result is likely to be license plate
        self.__confidence = 0.5
        
        # alpha paramater during normalization
        self.__alpha = 7.75 
        
        # confidence level of whatever the licence plate is one line high or two
        self.__plate_confidence = 1.4
        
        # output size (width, height) for one line high license plate
        self.__one_line = (400, 100) 
        
        # output size (width, height) for two lines high license plate
        self.__two_lines = (300, 300) 
        
        # loss function denoted the corresponding vertices of a canonical unit square centered at the origin
        self.__loss = np.matrix([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]).T
        
        # ----------
        # object detection part execution
        self.detect()
        # ----------
        
    @property
    def source(self):
        return self.__source
        
    @property
    def width(self):
        return self.__width
        
    @property
    def height(self):
        return self.__height
        
    @property
    def Y(self):
        return self._Y
        
    @Y.setter
    def Y(self, val):
        self._Y = val
  
    @property
    def net_path(self):
        try: 
            if readable(''.join([self.__net_path, '.h5'])) and readable(''.join([self.__net_path, '.json'])):
                return self.__net_path 
                
            else:
                raise Exception('netpath is not readable')
        except FileNotFoundError:
            return
    
    @property
    def iou_threshold(self):
        return self.__iou_threshold
        
    @property
    def confidence(self):
        return self.__confidence
        
    @property
    def alpha(self):
        return self.__alpha
                
    @property
    def plate_confidence(self):
        return self.__plate_confidence
        
    @property
    def one_line(self):
        return self.__one_line
            
    @property
    def two_lines(self):
        return self.__two_lines
            
    @property
    def loss(self):
        return self.__loss
        
    def preprocess_image(self):
        '''
            Turn an image into a matrix.
            
            Raises:
                Exception: the following source is not readable from the program side.
            
            Returns:
                Matrix with specific shape and values range
        '''
        
        if not readable(self.image_path):
            raise Exception('image_path is not readable or does not exist')

        img = cv2.imread(self.image_path) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255
        if img.shape[:2] != (self.width, self.height):
            img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))      
        return img
        
    def video(self):
        '''
            Catch the frame from a given optical source 
            and save it as an image file in the catalog of main file execution.
            
            Raises:
                Exception: If given optical sources cannot be opened.
        '''
        cap = cv2.VideoCapture(self.source)
        
        if not cap.isOpened():
            raise Exception("Could not open video device")
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        ret, frame = cap.read()

        cv2.imwrite(self.image_path, frame)
        cap.release()
        
    def load_model(self):
        '''
            Load the wpod_net to 
            predict an input matrix source.
            
            Returns:
                wpod_net with pre-trained weights
        '''

        json_file = open(pkg_resources.resource_filename(__name__, '/'.join(('wpod_net', 'wpod_net_update1.json'))), 'r') if self.net_path is None else open('{}.json'.format(self.net_path), 'r')
        model = model_from_json(json_file.read(), custom_objects={})
        model.load_weights(pkg_resources.resource_filename(__name__, '/'.join(('wpod_net', 'wpod_net_update1.h5'))) if self.net_path is None else '{}.h5'.format(self.net_path)) 
        json_file.close()
        return model
        
    @staticmethod
    def iou(tl1, br1, tl2, br2):
        '''
            "Intersection over Union" function implementation
            used by Non-max suppression
            
            Args:
                tl1: top left corner from the first box
                br1: bottom right corner from the first box
                tl2: top left corner from the second box
                br2: bottom right corner from the second box
                
            Returns:
                IOU value which determines the degree of overlapping boxes
        '''
        xi1 = max(tl1[0], tl2[0])
        yi1 = max(tl1[1], tl2[1])
        xi2 = min(br1[0], br2[0])
        yi2 = min(br1[1], br2[1])
        
        inter_area = (xi2 - xi1) * (yi2 - yi1)

        box1_area = (br1[1] - tl1[1]) * (br1[0] - tl1[0])
        box2_area = (br2[1] - tl2[1]) * (br2[0] - tl2[0])
        
        union_area = (box1_area + box2_area) - inter_area
        
        iou = inter_area / union_area
        return iou

    def nms(self):
        '''
           Filter overlapping boxes and select the most optimal ones from labels.
           
           Returns:
               None
        '''
        for idx, box in self.labels.get().copy().items():
            for i in range(idx+1):
                if self.iou(box['tl'], box['br'], self.labels[i]['tl'], self.labels[i]['br']) > self.iou_threshold:
                    self.labels.labels = {k: self.labels[k] for k in range(i+1)} 
                    return
        
    @staticmethod
    def find_T_matrix(pts, t_pts):
        '''
            Calculate A matrix using processed label and size points 
            for SVD.  
            
            Args:
                pts: the result of the ptsh function
                t_pts: the result of the draw_rectangle function
            
            Returns:
                The last row of SVD unitary matrix V*
        '''
        A = np.zeros((8, 9))
        
        for i in range(0, 4):
            xi = pts[:, i].T
            xil = t_pts[:, i]

            A[i*2, 3:6] = -xi
            A[i*2, 6:] = xil[1]*xi
            A[i*2+1, :3] = xi
            A[i*2+1, 6:] = -xil[0]*xi
            
        [U, E, V] = np.linalg.svd(A)
        T = V[-1, :].reshape((3, 3))
        
        return T

    def normalization(self, pts, mn, MN):
        '''
            Function required to match the network output resolution
            after scaling and re-centering according to each point (m, n)
            in the feature map.
            
            Args:
                pts: points of the propagated matrix by a loss function
                mn: point cell of the feature map
                MN: merged feature map volume
                
            Returns:
                Normalized matrix
        '''
        return ((pts * self.alpha) + mn) / MN
     
    def probs(self):
        '''
            Returns:
                The Probability that the given object is a license plate 
                in specific coordinate.
        '''
        return self.Y[..., 0]
        
    def affines(self):
        '''
            Returns:
                Affines from the specified coordinate.
        '''
        return self.Y[..., 2:]
        
    def draw_rectangle(self):
        '''
            Transform size indicators to get (4, 3) shape
            for T matrix transformer.
            
            Returns:
                Size as a matrix which 
                draws the rectangle on the plate.
        '''
        return np.matrix([[0, self.size[0], self.size[0], 0], [0, 0, self.size[1], self.size[1]], [1, 1, 1, 1]], dtype=float)
        
    def plate_coordinates(self):
        '''
            Get the aspect ratio of division between top left and bottom right
            coordinates to know the type of license plate.
            
            Returns:
                Size more likely to be two lines plate or one line plate
                concerning the confidence level
        '''
        size = (self.labels.get()[0]['tl'] - self.labels.get()[0]['br'])[0] / (self.labels.get()[0]['tl'] - self.labels.get()[0]['br'])[1]
        return self.two_lines if size < self.plate_confidence else self.one_line
                       
    def ptsh(self, point):
        '''
            Transform label points with the matrix shape 
            to get (4, 3) shape for T matrix transformer. 
            
            Args:
                point: Label point values
                
            Returns:
                The transformable array of label points
        '''
        return np.concatenate((point * np.array(self.X.shape[1::-1]).astype(float).reshape((2, 1)), np.ones((1, 4))))

    def collect_labels(self, xx, yy, probs, affines):
        '''
            Creates, propagates, and normalizes affine points as a matrix.
            It determines points, plate edges, and probabilities from the matrix
            and save it as a label for extraction.
            
            Args:
                xx: x coordinate which passes the threshold to be the desired object
                yy: y coordinate which passes the threshold to be the desired object
                probs: probabilities to be an object or not
                affines: features for a given point
        '''
        for idx, val in enumerate(xx):
            x, y = xx[idx], yy[idx]
            
            affine = affines[x, y]
            prob = probs[x, y]
            
            A = affine[[0, 1, 3, 4]].reshape((2, 2))
            A[0, 0] = max(A[0, 0], 0)
            A[1, 1] = max(A[1, 1], 0)

            pts = np.array(A*self.loss + affine[[2, 5]].reshape((2, 1)))

            MN = (np.array(self.X.shape[1::-1]).astype(float) / 16).reshape((2, 1)) #X
            mn = np.array([float(y) + 0.5, float(x) + 0.5]).reshape((2, 1))
            
            pts_prop = self.normalization(pts, mn, MN)
            
            self.labels[idx] = {'pts': pts_prop, 'tl': np.amin(pts, axis=1), 'br': np.amax(pts, axis=1), 'prob': prob}
          
    def pred(func):
        '''
            Wrapper for detect method
            
            Args:
                Function to be wrapped
            
            Returns:
                Prediction
        '''
        def get_Y(self):
            '''
                Predicts probabilities with affines
                on the X matrix using wpod_net model.
            '''
            if self.capture:
                self.video()
                
            self.X = self.preprocess_image()
            
            model = self.load_model()
            
            self.Y = model.predict(self.X)
            self.Y = np.squeeze(self.Y)
            func(self)
            
        return get_Y
        
    @pred
    def detect(self):
        '''
            LPD function to run.
            
            After prediction, it matches the corresponding points with affines
            and transfers it into the form of labels where each point is represented
            as transformed matrices with key point utilities.
            If the points are processed, it extracts the object from the X matrix.
        '''
        probs = self.probs()
        affines = self.affines()

        self.labels = SparseList()
        xx, yy = np.where(probs > self.confidence)
        self.collect_labels(xx, yy, probs, affines)
        self.nms()
        self.extract_plate()
        
            
    def extract_plate(self):
        '''
            With given final labels for matrices points, 
            warp perspective from T matrix and append it
            to license plate indicators.
            
            Raises:
                Exception: If no labels are returned from detection steps
        '''
        self.Y = []
        
        
        if self.labels.get():
            self.size = self.plate_coordinates()
            
            for label in self.labels.get().values():
                
                t_ptsh = self.draw_rectangle()
                ptsh = self.ptsh(label['pts'])
                
                T = self.find_T_matrix(ptsh, t_ptsh)

                self.Y = cv2.warpPerspective(self.X[0, ...], T, self.size)
        else:
            raise Exception('No plate detected')
            


class LPR(LPD):
    '''
    The second part of LPDR 
    
    With given settings:
        - Execute the LPD and acquire the storage for the next steps
        - The process resulting Y matrix to the better-performed form for OCR
        - Configure tesseract
        - Use tesseract to get the strings from the license plate and save it as a program output
    '''
    def __init__(self):
        super(LPR, self).__init__()
        
        ##### settings #####
        # default output value
        self.__output = None 
        
        # tesseract configuration setting for OCR Engine mode
        self.__oem = 3
        
        # tesseract configuration setting for running a subset of layout analysis
        self.__psm = 9 if self.size == self.one_line else 12
        
        
        # ---------
        # character recognition part execution
        self.recognize()
        # --------
        
    @property
    def oem(self):  
        if self.__oem in range(0, 4):    
            return self.__oem 
        else: 
            raise Exception("OEM Option not available for tesseract")

    @property
    def psm(self):
        if self.__psm in range(0, 14):    
            return self.__psm 
        else: 
            raise Exception("PSM Option not available for tesseract")
        
    @property
    def output(self):
        return self.__output 

    @output.setter
    def output(self, text):
        self.__output = ''.join(filter(str.isalnum, text))
        self.__output = ''.join(letter for letter in self.__output if (letter.isupper() or letter.isdigit()))
        
    def tesseract_conf(self):
        '''
            Transform tesseract settings as a
            executable configuration options.
        '''
        return r'--oem {} --psm {}'.format(self.oem, self.psm)
        
    def processing(func):
        '''
            Wrapper for recognizing method
            
            Args:
                Function to be wrapped
            
            Returns:
                Processing function
        '''
        def after_processing(self):
            '''
                Executing steps to be processed before OCR
            '''
            self.Y = self.rescale()
            self.Y = self.denoise()
            self.Y = self.erode()
            
            func(self)
            
        return after_processing
    
        
    def rescale(self):
        '''
            Returns:
                Rescaled and converted output matrix.
        '''
        return (self.Y * 255).astype('uint8')

    def denoise(self):
        '''
            Returns:
                Denoised output matrix.
        '''
        return cv2.fastNlMeansDenoisingColored(self.Y, None, 17, 29)

    def erode(self):
        '''
            Returns:
                Letter erosion effect on the output matrix.
        '''
        kernel = np.ones((5, 5), np.uint8)
        return cv2.erode(self.Y, kernel, iterations=1)
    
    @processing
    def recognize(self):
        '''
            Do the OCR operation on processed output by tesseract with a given configuration
            and save it as an output string which is the cumulative program result.   
        ''' 
        self.output = image_to_string(self.Y, config=self.tesseract_conf())
    

if __name__ == '__main__':
    print(LPR().output)

