from multiprocessing import Process
import time

def a():
    time.sleep(0.1)
    print 'hello'

p = Process(target=a)

p.start()

while p.is_alive():
    print p.is_alive()

print p.is_alive()
