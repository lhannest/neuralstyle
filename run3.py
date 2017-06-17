from styletransfer.model import buildModel as buildModel
from image import load_images, preprocess, combine_horizontally
from utilities import Printer, Timer, getTemp
import time
from multiprocessing import Process
import numpy as np
import scipy
import matplotlib.image as im
from image import deprocess

# Setting up parameters
content_image_path = 'images/2017/bia_art.jpg'
style_image_path = 'images/2017/oil_face_600.jpg'
save_path = 'images/sean/bia/'

# Loading and preprocessing the style and content images
content_image, style_image = load_images(content_image_path, style_image_path)

im.imsave(save_path + 'result[-1].jpg', combine_horizontally(content_image, style_image))

content_image, mean = preprocess(content_image)
style_image, _, = preprocess(style_image)

# Defining the loss and grad function
loss, grad = buildModel(content_image, style_image)

x = content_image.flatten()

im.imsave(save_path + 'result.jpg', deprocess(x.reshape(content_image.shape), content_image.shape, mean))

def func(x):
    """
    A function to be used by scipy.optimize.fmin_l_bfgs_b
    Take a look at the documentation for fmin_l_bfgs_b to see why this is useful
    """
    x = np.array(x, dtype='float64').flatten()
    return loss(x), np.asarray(grad(x), dtype='float64')

class Callback(object):
    def __init__(self):
        self.i = 0
    def __call__(self, xk=None):
        self.i += 1
        print 'iteration', self.i, str(getTemp()) + " C", loss(xk)
        # if self.i % 5 == 0:
        im.imsave(save_path + 'result[' + str(self.i) + '].jpg', deprocess(xk, content_image.shape, mean))
        if getTemp() > 70:
            while getTemp() > 70:
                print 'waiting for temperature to drop below 60C, it is currently at ' + str(getTemp()) + 'C'
                time.sleep(5)

from scipy.optimize import fmin_l_bfgs_b

print "starting with loss:", loss(x)
x, f, d = fmin_l_bfgs_b(func, x, callback=Callback())
print "ending with loss:  ", loss(x)
print 'iterations:', d['nit']
print 'function calls:', d['funcalls']
print 'warn flag:', d['warnflag']


im.imsave(save_path + 'final.jpg', deprocess(x, content_image.shape, mean))
