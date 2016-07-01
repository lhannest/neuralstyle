import theano
import theano.tensor as T

import lasagne

from models.vgg19 import build_model

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import scipy

def getImage(path, resize_to=None):
    """Loads, resizes and preprocesses the image, returning a 4D tensor."""
    if resize_to == None:
        img = Image.open(path)
    else:
        img = Image.open(path).resize(resize_to)
    img = np.asarray(img).transpose(2, 0, 1)
    img = np.reshape(img, (1,) + img.shape)
    mean = np.mean(img)
    img = img - mean
    return img, mean

def deprocess(img, mean):
    """deprocesses image so that it can be desplayed by plt.imshow()"""
    img = np.asarray(img)
    img = img + mean
    img = img.reshape(img.shape[1:])
    img = img.transpose(1, 2, 0)
    return np.clip(img, 0, 255).astype('uint8')

# Loading and preprocessing the images, and setting up the theano symbolic variables
content_img, content_mean = getImage('images/photo.jpg')
style_img, style_mean = getImage('images/art1.jpg')

# Initializing theano shared variables for the content and style images
CON = theano.shared(np.asarray(content_img, dtype=theano.config.floatX))
STY = theano.shared(np.asarray(style_img, dtype=theano.config.floatX))

# scipy.optimize.fmin_l_bfgs_b will want a flattened array for the input of the function to be optimized
X = T.vector()
GEN = T.reshape(X, content_img.shape)

# Here we build the VGG-19 neural network, and get the filter responses of each image from the designated layers (not all will be used)
model = build_model()
layer_names = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

# We're flattening each chanel of the output of the VGG-19
CON_F = {name: T.flatten(lasagne.layers.get_output(model[name], inputs=CON), ndim=3) for name in layer_names}
STY_F = {name: T.flatten(lasagne.layers.get_output(model[name], inputs=STY), ndim=3) for name in layer_names}
GEN_F = {name: T.flatten(lasagne.layers.get_output(model[name], inputs=GEN), ndim=3) for name in layer_names}

# Evaluate those filter responses and save as theano shared variables (so that we don't have to evaluate them over and over)
CON_F = {name: theano.shared(F.eval()) for name, F in CON_F.items()}
STY_F = {name: theano.shared(F.eval()) for name, F in STY_F.items()}

def content_loss(A, B):
    # Returns the sum of squared errors of A and B
    return T.sum(T.sqr(A - B)) / 2.

def gram_matrix(M):
    # Converts M from shape (1, num filters, width, height) to (1, num filters, width * height)
    M = M.flatten(ndim=3)
    # Returns the graham matrix of M, over the third axis
    return T.tensordot(M, M, axes=[2, 2])

def style_loss(A, B):
    gram_A = gram_matrix(A)
    gram_B = gram_matrix(B)
    # The number of filters
    N = A.shape[1]
    # The number of elements (width * height)
    M = A.shape[2] * A.shape[3]
    return  (1. / 4 * N**2 * M**2) * T.sum(T.sqr(gram_A - gram_B))

content_layers = ['conv5_1']
style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

content_losses = [content_loss(GEN_F[layer], CON_F[layer]) for layer in content_layers]
style_losses = [1e6 * style_loss(GEN_F[layer], STY_F[layer]) for i, layer in enumerate(style_layers)]

alpha = 0.1e-4
beta = 0.1
total_loss = alpha * sum(content_losses) + beta * sum(style_losses) / 1. * len(style_losses)

loss = theano.function(inputs=[X], outputs=total_loss, allow_input_downcast=True)
grad = theano.function(inputs=[X], outputs=T.grad(total_loss, wrt=X), allow_input_downcast=True)

images = []
x = content_img.flatten()

def func(x):
    """
    A function to be used by scipy.optimize.fmin_l_bfgs_b.
    Take a look at the documentation for fmin_l_bfgs_b to see why this is useful.
    """
    x = np.array(x, dtype='float64').flatten()
    return loss(x), np.asarray(grad(x), dtype='float64')

from subprocess import Popen, PIPE
def getTemp():
	cmd = ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader']
	p = Popen(cmd, stdout=PIPE)
	out, err = p.communicate()
	return out[:-1]

t0 = 0
images = []
for i in range(5):
    wait_time = t0
    print 'waiting for ', wait_time, 'which is', timesince(wait_time)
    time.sleep(wait_time)
    print 'starting', i
    t = time.time()
    x, f, d = scipy.optimize.fmin_l_bfgs_b(func, x, maxfun=15)
    t0 = time.time() - t
    print timesince(t), getTemp() + 'C'
    images.append(deprocess(np.asarray(x).reshape(content_img.shape), content_mean))
x = np.asarray(x).reshape(content_img.shape)
plt.imshow(deprocess(x, content_mean))
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.show()
import plot
plot.plot_all(images, 2, 5)
