import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import matplotlib.image as im
import numpy as np
import lasagne

from datetime import datetime

import image
import vgg19
import adam

from scipy.optimize import fmin_l_bfgs_b

def imshow(img):
    plt.imshow(img)
    plt.show()

def getLayers(img, model, layer_names):
    return [lasagne.layers.get_output(model[layer], inputs=img).flatten(ndim=3) for layer in layer_names]

def setup_methods(content_img, style_img):
    """
    A layer with N distinct filters has N feature maps, each of size M where M
    is the height times the width of the feature map.
    So the response of a layer can be stored in an NxM matrix called F, having
    N rows and M columns. Each row of F represents a flattened feature map of
    the layer. And F[i,j] is the activation of the ith filter at position j
    in the layer, and we define F to be the content representation for that
    layer.

    content_loss(img, layer) = ...

    """

    a = theano.shared(np.asarray(content_img, dtype=theano.config.floatX))
    p = theano.shared(np.asarray(style_img, dtype=theano.config.floatX))
    x = theano.shared(np.asarray(content_img, dtype=theano.config.floatX))
    # X = T.tensor4()
    # x = X

    content_layers = ['conv5_1', 'conv4_1']
    style_layers = ['conv1_1', 'conv2_1', 'conv3_1']

    model = vgg19.build_model(content_img.shape) #content_img.shape[2], content_img.shape[3])

    P = getLayers(p, model, content_layers)
    F = getLayers(x, model, content_layers)

    # Calculate once and re-use
    P = [theano.shared(p.eval()) for p in P]

    contentLoss = sum([0.5 * T.sum(T.sqr(Fl - Pl)) for Fl, Pl in zip(F, P)])

    A = getLayers(a, model, style_layers)
    F = getLayers(x, model, style_layers)

    G = [T.tensordot(u, u, axes=[2, 2]) for u in F]
    A = [T.tensordot(u, u, axes=[2, 2]) for u in A]

    # Calculate once and re-use
    A = [theano.shared(a.eval()) for a in A]

    N = A[0].shape[1]
    M = A[0].shape[2]

    E = [(1. / 4. * N**2 * M**2) * T.sum(T.sqr(u - v)) for u, v in zip(G, A)]

    styleLoss = T.sum(E) / (1. * len(G))

    alpha, beta = 10**-3, 1

    totalLoss = alpha * styleLoss + beta * contentLoss

    # loss = theano.function(inputs=[X], outputs=totalLoss, allow_input_downcast=True)
    # grad = theano.function(inputs=[X], outputs=T.grad(totalLoss, wrt=X), allow_input_downcast=True)

    updates = adam.Adam(cost=totalLoss, params=[x])
    adam_optimize = theano.function(inputs=[], outputs=totalLoss, updates=updates, allow_input_downcast=True)
    getImage = theano.function(inputs=[], outputs=x)
    getLoss = theano.function(inputs=[], outputs=totalLoss)
    # return adam_optimize, getImage

    tensor = x
    def optimize():
        tensor, f, d = fmin_l_bfgs_b(loss, tensor, grad)
        return x, f, d['grad']

    return optimize

def image_to_tensor4(img):
    img = np.asarray(img, dtype=theano.config.floatX)
    s1 = img.shape
    img = np.transpose(img, [2, 0, 1])
    img = np.reshape(img, (1,) + img.shape)
    s2 = img.shape
    print 'image to tensor:', s1, '->', s2
    return img

def tensor4_to_image(tensor4):
    tensor4 = np.asarray(tensor4)
    s1 = tensor4.shape
    tensor4 = np.reshape(tensor4, tensor4.shape[1:])
    tensor4 = np.transpose(tensor4, [1, 2, 0])
    s2 = tensor4.shape
    print 'tensor to image:', s1, '->', s2
    tensor4 = np.clip(tensor4, 0, 255).astype('uint8')
    return tensor4

if __name__ == '__main__':
    print str(datetime.now())
    content_image_path = '../images/big_photo.jpg'
    style_image_path = '../images/big_art.jpg'
    save_path = '../images/sean/'
    num_iterations = 200

    photo, art = image.load_images(content_image_path, style_image_path)

    im.imsave(save_path + 'result[-1].jpg', image.combine_horizontally(photo, art))

    photo_tensor = image_to_tensor4(photo)
    art_tensor = image_to_tensor4(art)

    adam_optimize, getImage = setup_methods(photo_tensor, art_tensor)

    x = photo_tensor
    for i in range(num_iterations):
        print i, str(datetime.now())
        adam_optimize()

        im.imsave(save_path + 'result[' + str(i) + '].jpg', tensor4_to_image(getImage()))

# TODO:
#     - make the G and P shared variables, so they don't need to be calculated
#       over and over.
