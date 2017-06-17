import theano
import theano.tensor as T
from theano.tensor.extra_ops import to_one_hot
import numpy as np
from sklearn.datasets import fetch_mldata

mnist = fetch_mldata('MNIST original')

print(mnist.data.shape)
print(mnist.target.shape)


x = T.vector()
t = T.ivector()
onehot = to_one_hot(t, 10);

w = theano.shared(np.random.randn(784, 10))

y = T.dot(x, w)

mse = T.mean(T.sqrt(y - onehot))

updates = [(w, w - T.grad(cost=mse, wrt=w))]

evaluate = theano.function(inputs=[x], outputs=T.argmax(y))
train = theano.function(inputs=[x, t], outputs=mse, updates=updates)

for img, label in zip(mnist.data, mnist.target):
    print(img.shape, label, train(img, [int(label)]))
