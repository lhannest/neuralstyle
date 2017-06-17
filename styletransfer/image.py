from PIL import Image
import numpy as np

def load_images(content_image_path, style_image_path):
    content_image = Image.open(content_image_path)
    style_image = Image.open(style_image_path)

    width = min(content_image.size[0], style_image.size[0])
    height = min(content_image.size[1], style_image.size[1])

    content_image = crop(content_image, width, height)
    style_image = crop(style_image, width, height)

    return content_image, style_image

def crop(image, width, height):
    old_width, old_height = image.size
    left = (old_width - width) / 2
    right = (old_width + width) / 2
    top = (old_height - height) / 2
    bottom = (old_height + height) / 2
    return image.crop((left, top, right, bottom))

"""
    Note that preprocess and deprocess are not inverse methods of each other.

    When PIL.Image loads an image from file, it is of shape (width, height, 3),
    but when upon passing it into the convolutional network Theano expects it to
    be of shape (1, 3, width, height). Then we flatten the last two dimensions
    together to build the loss functions, so it becomes of shape
    (1, 3, width * height). Finally, scipy.optimize.fmin_l_bfgs_b expects the
    image to be of shape (3*width*height,), i.e., a 1d vector rather than
    3d matrix or 4d tensor.

    As such, preprocess takes a PIL.Image object and converts it into a 4d
    tensor for Theano. Then, deprocess takes a vector and converts it back into
    a 3d matrix to be displayed or saved as an image.
"""

def preprocess(image):
    image = np.asarray(image)
    # When loaded by PIL, image is of shape (width, height, 3). We now transpose
    # the axes so that its shape is (3, width, height).
    image = np.asarray(image).transpose(2, 0, 1)
    # We now reshape the image so that it is shape is (1, 3, width height). It
    # is now a 4d tensor, which is what theano's convolution opperator wants as
    # an input.
    image = np.reshape(image, (1,) + image.shape)
    # We now set the mean of the image to zero, and return it so it can be added
    # back into the resulting image later.
    mean = np.mean(image)
    image = image - mean
    return image, mean

def deprocess(vectorized_image, shape, mean):
    img = np.asarray(vectorized_image).reshape(shape)
    img = img + mean
    img = img.reshape(img.shape[1:])
    img = img.transpose(1, 2, 0)
    return np.clip(img, 0, 255).astype('uint8')

def combine_horizontally(*images):
    """
    each image in images must be the same size
    """
    images = map(np.asarray, images)
    return np.hstack(images)

if __name__ == '__main__':
    photo, art = load_images('images/photo1.jpg', 'images/art1.jpg')

    print photo.size, art.size

# def getImage(path, resize_to=None):
#     """Loads, resizes and preprocesses the image, returning a 4D tensor."""
#     if resize_to == None:
#         img = Image.open(path)
#     else:
#         img = Image.open(path).resize(resize_to)
#     img = np.asarray(img).transpose(2, 0, 1)
#     img = np.reshape(img, (1,) + img.shape)
#     mean = np.mean(img)
#     img = img - mean
#     return img, mean

# def deprocess(tensor4d, mean):
#     """deprocesses the 4D tensor representing an image so that it can be properly desplayed"""
#     img = np.asarray(tensor4d)
#     img = img + mean
#     img = img.reshape(img.shape[1:])
#     img = img.transpose(1, 2, 0)
#     return np.clip(img, 0, 255).astype('uint8')


# # Loading and preprocessing the images, and setting up the theano symbolic variables
# content_img, content_mean = getImage('images/big_photo.jpg')
# style_img, style_mean = getImage('images/big_art.jpg')
#
# print content_img.shape
# print style_img.shape
