import theano
import theano.tensor as T

import lasagne

from styletransfer.vgg19 import build_model

import matplotlib.pyplot as plt
import matplotlib.image as image
import numpy as np
import scipy

def setup_methods(content_img, style_img):
    # Initializing theano shared variables for the content and style images
    CON = theano.shared(np.asarray(content_img, dtype=theano.config.floatX))
    STY = theano.shared(np.asarray(style_img, dtype=theano.config.floatX))

    # scipy.optimize.fmin_l_bfgs_b will want a flattened array for the input of the function to be optimized
    X = T.vector()
    GEN = T.reshape(X, content_img.shape)

    # Here we build the VGG-19 neural network, and get the filter responses of each image from the designated layers (not all will be used)
    model = build_model()#content_img.shape[2], content_img.shape[3])

    # Here we choose which layers of the VGG-19 neural network will be used for the style and content error
    content_layers = ['conv5_1', 'conv4_1']
    style_layers = ['conv1_1', 'conv2_1', 'conv3_1']

    # f = open("images/aug/info.txt", "w")
    # f.write("content layers: " + str(content_layers) + "\nstyle layers: " + str(style_layers))
    # f.close()

    # Don't want to allow for duplicates
    layer_names = set(content_layers + style_layers)

    CON_F = {name: lasagne.layers.get_output(model[name], inputs=CON).flatten(ndim=3) for name in layer_names}
    STY_F = {name: lasagne.layers.get_output(model[name], inputs=STY).flatten(ndim=3) for name in layer_names}
    GEN_F = {name: lasagne.layers.get_output(model[name], inputs=GEN).flatten(ndim=3) for name in layer_names}

    # Evaluate those filter responses and save as theano shared variables (so that we don't have to evaluate them over and over)
    CON_F = {name: theano.shared(F.eval()) for name, F in CON_F.items()}
    STY_F = {name: theano.shared(F.eval()) for name, F in STY_F.items()}

    def content_loss(A, B):
        # Returns the sum of squared errors of A and B
        return T.sum(T.sqr(A - B)) / 2.

    def style_loss(A, B):
        gram_A = T.tensordot(A, A, axes=[2, 2])
        gram_B = T.tensordot(B, B, axes=[2, 2])
        # The number of filters
        N = A.shape[1]
        # The number of elements (width * height)
        M = A.shape[2]
        return  (1. / 4 * N**2 * M**2) * T.sum(T.sqr(gram_A - gram_B))

    content_losses = [content_loss(GEN_F[layer], CON_F[layer]) for layer in content_layers]
    style_losses = [style_loss(GEN_F[layer], STY_F[layer]) for i, layer in enumerate(style_layers)]

    alpha = 0.001e-4
    beta = 0.001
    total_loss = alpha * sum(content_losses) + beta * sum(style_losses) / 1. * len(style_losses)

    loss = theano.function(inputs=[X], outputs=total_loss, allow_input_downcast=True)
    grad = theano.function(inputs=[X], outputs=T.grad(total_loss, wrt=X), allow_input_downcast=True)

    return loss, grad

from image import load_images
def run(content_image, style_image):
    x = content_img.flatten()

    def func(x):
        """
        A function to be used by scipy.optimize.fmin_l_bfgs_b.
        Take a look at the documentation for fmin_l_bfgs_b to see why this is useful.
        """
        x = np.array(x, dtype='float64').flatten()
        return loss(x), np.asarray(grad(x), dtype='float64')

    from utilities import Printer, Timer, getTemp
    p = Printer(0.1)
    timer = Timer()
    t0 = 0
    images = []
    ITERATIONS = 5
    for i in range(ITERATIONS):
        # timer.setMarker('temp_wait')
        # while getTemp() > 55:
        #     p.overwrite('Temperature is ' + str(getTemp()) + 'C, waiting for it to be 55C -- ' + timer.timeSince('temp_wait'))
        #     timer.sleep(2)
        # p.clear()

        print('starting', i)
        timer.setMarker('optimize')
        x, f, d = scipy.optimize.fmin_l_bfgs_b(func, x, maxfun=15)
        print(timer.timeSince('optimize'), str(getTemp()) + 'C')
        # images.append(deprocess(np.asarray(x).reshape(content_img.shape), content_mean))
        # image.imsave('images/aug/imsave[' + str(i) + '].png', deprocess(np.asarray(x).reshape(content_img.shape), content_mean))


# x = np.asarray(x).reshape(content_img.shape)
# plt.imshow(deprocess(x, content_mean))
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.show()
# plot.plot_all(images, 2, int(ITERATIONS/1.5))
