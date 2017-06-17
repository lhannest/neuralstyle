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
