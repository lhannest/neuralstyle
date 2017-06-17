import numpy as np
from scipy.optimize import minimize, rosen
import time
import warnings

class TookTooLong(Warning):
    pass

class MinimizeStopper(object):
    def __init__(self, max_sec=60):
        self.max_sec = max_sec
        self.start = time.time()
    def __call__(self, xk=None):
        elapsed = time.time() - self.start
        if elapsed > self.max_sec:
            warnings.warn("Terminating optimization: time limit reached",
                          TookTooLong)
        else:
            # you might want to report other stuff here
            print("Elapsed: %.3f sec" % elapsed)

class Callback(object):
    def __init__(self):
        self.i = 0
    def __call__(self, xk=None):
        self.i += 1
        print self.i
        if self.i == 10:
            print "Sleeping..."
            time.sleep(5)
            print "Awoken"

# example usage
x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
res = minimize(rosen, x0, method='L-BFGS-B', callback=MinimizeStopper(0.001))
print "Not finished!"
