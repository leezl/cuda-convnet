'''
Tests for the camera streamer
'''

import random
import unittest
import time
from cameraStreamer import CameraStreamer

class TestCameraStreamer(unittest.TestCase):

    def setUp(self):
        batch_size = 2
        self.streamer = CameraStreamer(batchSize=batch_size, size = 'quarter', max_lag=2)

    def test_images(self):
        self.streamer.begin_streaming()
        time.sleep(5)
        self.streamer.end_streaming()
        #check shapes of data sensible
        batch = self.streamer.get_next_batch()
        #check resolution or what...

    def test_timing(self):
        #check timing of two streaming methods
        firstLength=0
        secondLength=0
        firstLength2=0
        secondLength2=0
        # check both camera streamers and check that fast one is faster
        self.streamer.begin_streaming(True)
        #wait for some images to accumulate
        time.sleep(5)
        self.streamer.end_streaming()
        firstLength = self.streamer.clear_queue()
        #and second half
        time.sleep(2)
        self.streamer.begin_streaming(False)
        #wait for some images to accumulate
        time.sleep(5)
        self.streamer.end_streaming()
        secondLength = self.streamer.clear_queue()

        time.sleep(2)
        #and other order
        self.streamer.begin_streaming(False)
        #wait for some images to accumulate
        time.sleep(5)
        self.streamer.end_streaming()
        firstLength2 = self.streamer.clear_queue()
        time.sleep(2)
        self.streamer.begin_streaming(True)
        #wait for some images to accumulate
        time.sleep(5)
        self.streamer.end_streaming()
        secondLength2 = self.streamer.clear_queue()

        print "Fast Method: ",firstLength,', Slow Method: ',secondLength
        print "Second pass Fast ",firstLength2,', Slow ',secondLength2s
        assert firstLength2>secondLength2, "Faster method is not faster..."

if __name__ == '__main__':
    unittest.main()