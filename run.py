from neuralstyle import setup_methods
from image import load_images, preprocess, combine_horizontally
from utilities import Printer, Timer, getTemp
from multiprocessing import Process
import numpy as np
import scipy
import matplotlib.image as im
from image import deprocess

from styletransfer.model import buildModel

# Setting up parameters
content_image_path = 'images/big_photo.jpg'
style_image_path = 'images/big_art.jpg'
save_path = 'images/sean/'
num_iterations = 5

# Loading and preprocessing the style and content images
content_image, style_image = load_images(content_image_path, style_image_path)

im.imsave(save_path + 'result[-1].jpg', combine_horizontally(content_image, style_image))

content_image, mean = preprocess(content_image)
style_image, _, = preprocess(style_image)

# Defining the loss and grad function
loss, grad = buildModel(content_image, style_image)

# This method displays the time on loop, we will open a new process to run it
# so it can be printing to the screen while fmin_l_bfgs_b runs.
def text_display(i, printer, timer, getTemp):
    while True:
        tmp = getTemp()
        time = timer.timeSince('optimizing')
        msg = 'Working on iteration ' + str(i) + ', has been taking: ' + time + ', GPU temperature is ' + str(tmp) + 'C'
        printer.overwrite(msg)

def cool_down():
    while getTemp() > 55:
        printer.overwrite('GPU temperature is ' + str(getTemp()) + 'C, waiting for it to be 55C')
        timer.sleep(2)

class Callback:
    def __init__(self, *argv):
        self.i = 0
    def __call__(self, *argv):
        self.i += 1
        print '[' + str(self.i) + ']'

x = content_image.flatten()

printer = Printer(0.5)
timer = Timer()
for i in range(num_iterations):
    cool_down()

    timer.setMarker('optimizing')
    p = Process(target=text_display, args=(i, printer, timer, getTemp))
    p.start()

    def func(x):
        """
        A function to be used by scipy.optimize.fmin_l_bfgs_b
        Take a look at the documentation for fmin_l_bfgs_b to see why this is useful
        """
        x = np.array(x, dtype='float64').flatten()
        return loss(x), np.asarray(grad(x), dtype='float64')

    x, f, d = scipy.optimize.fmin_l_bfgs_b(func, x, maxfun=1, callback=Callback())
    print
    print loss(x)
    print
    p.terminate()
    printer.clear()

    print 'Iteration', i, 'took ' + timer.timeSince('optimizing'), 'finished with a GPU temperature of ' + str(getTemp()) + 'C'

    im.imsave(save_path + 'result[' + str(i) + '].jpg', deprocess(x, content_image.shape, mean))
