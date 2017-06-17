import sys
import time
import datetime

from subprocess import Popen, PIPE
def getTemp():
	cmd = ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader']
	p = Popen(cmd, stdout=PIPE)
	out, err = p.communicate()
	return int(out[:-1])

class Timer(object):
    def __init__(self):
        self.markers = {}
    def setMarker(self, marker_name):
        self.markers[marker_name] = time.time()
    def timeSince(self, marker_name):
        t = int(time.time() - self.markers[marker_name])
        return str(datetime.timedelta(seconds=t))
    def sleep(self, t):
        time.sleep(t)


def gettime():
    """
    Wrapper for time.time()
    """
    return time.time()

def timesince(t):
    """
    Returns a string representing t seconds as hours, minuits and seconds.
    """
    return str(datetime.timedelta(seconds=t))


class Printer(object):
	def __init__(self, wait_time):
		self.wait_time = wait_time
		self.t = time.time()
		self.last_length = 0
	def overwrite(self, message='', wait=True):
		if time.time() - self.t >= self.wait_time or not wait:
			sys.stdout.write('\r' + ' '*self.last_length + '\r')
			sys.stdout.flush()
			sys.stdout.write(message)
			sys.stdout.flush()
			self.t = time.time()
			self.last_length = len(message)
	def clear(self):
		self.overwrite(wait=False)
