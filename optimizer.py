import numpy as np
from scipy.optimize import fmin_l_bfgs_b

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

class Optimizer(object):
    def __init__(self, loss, grad):
        self.loss = loss
        self.grad = grad

    def optimize(self, x):
        try:
            # x, f, d = fmin_l_bfgs_b(func, x, callback=Callback(), maxfun=1)
            s = [d / 100 * 2 for d in x.shape]

            return x
        except KeyboardInterrupt:
            print 'Interrupted'

if __name__ == '__main__':
    x = np.random.randn(5, 5)

    w = x.shape[1]
    w = 2 * (w / 3)

    

    print w
    # opt = Optimizer(None, None)
    # print opt.optimize(x)
