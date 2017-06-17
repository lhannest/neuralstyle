# -*- coding: utf-8 -*-
"""
content_tensor and style_tensor must be 4d tensors representing the content
and style source images. Their shape must be (1, 3, width, height), with
them both having the same width and height. Typically there will be three colour
chanels, that is where the 3 comes from.

A layer with N distinct filters has N feature maps, each of size M where M
is the height times the width of the feature map.
So the response of a layer can be stored in an NxM matrix called F, having
N rows and M columns. Each row of F represents a flattened feature map of
the layer. And F[i,j] is the activation of the ith filter at position j
in the flattened filter.

Here F is the feature representation of that layer.

"""
import theano
import theano.tensor as T
import numpy as np
import lasagne
import vgg19
from lasagne.utils import floatX

content_layers = ['conv5_1']
style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
alpha = 0.001e-4
beta = 0.001

model = vgg19.build_model()#tensor.shape)

def getFeatureRep(input_tensor, layer_name):
    """ Returns the feature representation F ∈ ℝ^(NxM) """
    output_tensor = lasagne.layers.get_output(model[layer_name], inputs=input_tensor)
    return output_tensor.flatten(ndim=3).dimshuffle([1, 2])

def getStyleRep(feature_representation):
    """ Returns the style representation G ∈ ℝ^(NxN) """
    return T.dot(feature_representation, T.transpose(feature_representation))

def contentLoss(image_tensor, content_tensor, layer_name):
    F = getFeatureRep(image_tensor, layer_name)
    P = getFeatureRep(content_tensor, layer_name)
    P = theano.shared(P.eval())
    loss = 0.5 * T.sum(T.sqr(F - P))

def styleLoss(image_feature, style_feature):
    None

def total_variation_loss(x):
    return (((x[:,:,:-1,:-1] - x[:,:,1:,:-1])**2 + (x[:,:,:-1,:-1] - x[:,:,:-1,1:])**2)**1.25).sum()

def buildModel(pic_tensor, art_tensor):
    pic_tensor = floatX(pic_tensor)
    art_tensor = floatX(art_tensor)

    # The original paper starts from random noise, but I find this works better
    X = T.vector()
    img_tensor = T.reshape(X, pic_tensor.shape)

    img_feature = [getFeatureRep(img_tensor, layer_name) for layer_name in content_layers]
    pic_feature = [getFeatureRep(pic_tensor, layer_name) for layer_name in content_layers]

    # We want to compile this result, so we don't have to calculate it over and over
    # pic_feature = [theano.shared(floatX(feature.eval())) for feature in pic_feature]

    contentLoss = sum([0.5 * T.sum(T.sqr(F - P)) for F, P in zip(img_feature, pic_feature)])

    # TODO: In the paper G is supposed to be dot(F, transpose(F))
    #       but in fact a gram matrix should be dot(transpose(F), F).

    img_feature = [getFeatureRep(img_tensor, layer_name) for layer_name in style_layers]
    art_feature = [getFeatureRep(art_tensor, layer_name) for layer_name in style_layers]

    # We want to compile this result, so we don't have to calculate it over and over
    # art_feature = [theano.shared(floatX(feature.eval())) for feature in art_feature]

    E = []

    for F, P in zip(img_feature, art_feature):
        N, M = P.shape.eval()
        G = getStyleRep(F)
        A = getStyleRep(P)
        e = T.sum(T.sqr(G - A)) * theano.shared(floatX((1. / 4. * N**2 * M**2)))
        E.append(e)

    styleLoss = sum(E) / len(E)

    totalLoss = (alpha * contentLoss) + (beta * styleLoss) + total_variation_loss(img_tensor)

    loss = theano.function(inputs=[X], outputs=totalLoss, allow_input_downcast=True)
    grad = theano.function(inputs=[X], outputs=T.grad(totalLoss, wrt=X), allow_input_downcast=True)

    return loss, grad
