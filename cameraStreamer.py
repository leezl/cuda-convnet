import sys, os, platform
import time
import threading
import Queue
import ctypes
from primesense import openni2
from primesense import _openni2 as c_api
#from primesense import nite
from data import *
import numpy.random as nr
import numpy as n
import random as r
#temporary for checking display:
#import matplotlib.pyplot as plt
import cv2

# add to sys.path?
print "Sys setup ",os.name,',',platform.system(),',',platform.release()
for item in sys.path:
    print item

if "ARCH" in platform.release():
    print "Running on Arch "
    sys.path.append(sys.path[0]+"/arch-linux")
else:
    sys.path.append(sys.path[0]+"/ubuntu-linux")
    
for item in sys.path:
    print item

'''
Potential Issue: 
Python Threading does not use multiple CPUs. So, not parallel, just switches.
'''

def show_image(image):
    '''if image.shape[2]==3:
        plt.imshow(image)
    else:
        plt.imshow(n.dstack((image,image,image)))
    plt.pause(.5)
    plt.draw()
    #plt.show()
    plt.close()'''
    cv2.imshow('Color',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    emergency_quit()

def emergency_quit():
    s = raw_input('--> ')
    if s=='q':
        sys.exit()

class CameraStreamer:
    '''
    init: sets all initial parameters: 
        how many images to collect, 
        camera selection, 
        camera parameters,
        initial streams,
        pixel format,
        syncing,
        registration, (currently broken)
    '''
    def __init__(self, batchSize, size='full', max_lag=2, max_size=40, wait_time=0.0, with_depth=True, mean_img=None):
        #storage for our frames from the camera
        self.batchQueue = Queue.Queue() #queue or list as queue? need threads? locks? to grab more then one item?
        #how far behind is the calling program allowed to be in batches (we can have x batches stashed before we start dropping them)
        self.max_batch_lag = 2
        #how many frames do we hand back each time we're asked
        self.batch_size = batchSize
        #are we currently reading from the device
        self.streaming = False
        #what resolution do we want to use
        self.size = size
        #decrease frame rate...bad idea, but let's make it possible
        self.pause_time = wait_time
        #only store this much data at once here...
        self.max_batch_size = max_size
        #stream both, or one
        self.both_streams = with_depth
        #depth type:
        self.depth_type = c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM
        self.color_type = c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888#ONI_PIXEL_FORMAT_JPEG
        if mean_img:
            self.mean_img=mean_img
        #self.depth_type = ONI_PIXEL_FORMAT_DEPTH_100_UM
        #only using quarter or full size of the standard resolution
        if size =='quarter':
            self.shape = (240, 320, 4)
        else:
            #print "Using full size first"
            self.shape = (480, 640, 4)#no idea why the printing prgram swaps these. but it does.
        print self.size,',',self.shape 
        #this is used for ending the thread that reads from the camera
        self.do_not_exit = True
        #set up the camera to resolutiona dn pixel with syncing and registration (if possible)
        self.setup_camera()

    '''
    setup_camera:
    init openni (so we can access cameras),
    get any device,
    set parameters of camera and streams,
    '''
    def setup_camera(self):
        #initialize the camera
        openni2.initialize(sys.path[-1])
        #AND/OR initialize niTE? (also inits openni)
        #find device
        self.device = openni2.Device.open_any()
        #maybe better error checker available
        if not self.device:
            print "Error finding device. Check OpenNi docs."
            sys.exit();

        #debugging: print the options: pixel types, resolutions
        #self.print_video_options(self.device, openni2.SENSOR_DEPTH)
        #self.print_video_options(self.device, openni2.SENSOR_COLOR)

        if self.both_streams:
            #make sure color and depth are synced, why would we not want this? If we're only grabbing one.
            if not self.device.get_depth_color_sync_enabled():
                self.device.set_depth_color_sync_enabled(True)

            #also: registration mode: could not find parameters that worked here...
            if self.device.get_depth_color_sync_enabled():
                print "Color-Depth Sync is enabled"
            else:
                print "Sync is disabled"
                self.device.set_depth_color_sync_enabled(True)

            #registration: not correctly implemented in python bindings, but if they fix it put it here

            #create both streams: perhaps this should be done later, close to Start of streaming
            self.depth_stream = self.device.create_depth_stream()

            #set pixel format and resolution for both streams
            if self.size == 'quarter':
                self.depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat = self.depth_type, resolutionX = 320, resolutionY = 240, fps = 30))
            elif self.size == 'full':
                self.depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat = self.depth_type, resolutionX = 640, resolutionY = 480, fps = 30))
            else:
                print "No recognizeable video resolution given, defaulting to QVGA ('quarter')"
                self.depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat = self.depth_type, resolutionX = 320, resolutionY = 240, fps = 30))

        #color only by default
        self.color_stream = self.device.create_color_stream()
        if self.size == 'quarter':
            self.color_stream.set_video_mode(c_api.OniVideoMode(pixelFormat = self.color_type, resolutionX = 320, resolutionY = 240, fps = 30))
        elif self.size == 'full':
            self.color_stream.set_video_mode(c_api.OniVideoMode(pixelFormat = self.color_type, resolutionX = 640, resolutionY = 480, fps = 30))
        else:
            print "No recognizeable video resolution given, defaulting to QVGA ('quarter')"
            self.color_stream.set_video_mode(c_api.OniVideoMode(pixelFormat = self.color_type, resolutionX = 320, resolutionY = 240, fps = 30))

    '''
    get_batch: 
        hands back batch from Queue
        checks on size available: probably need to fail on Queue Empty
    '''
    def get_next_batch(self):
        if self.batchQueue.qsize()<self.batch_size and not self.batchQueue.empty():
            more = self.batch_size-self.batchQueue.qsize()
            print "WARNING: Too few frames ready for next batch of ", self.batch_size,", filling with ",more, " duplicates. "
            #add more: this will not be good if we depend on time later...
            beginList = [self.batchQueue.get() for i in range(0, self.batchQueue.qsize())]
            #stretch the list out (duplicate frames evenly)
            print "stretching"
            self.stretch_to_other(beginList, len(beginList)+more)
            print "data as array"
            data = n.array(beginList)
        elif self.batchQueue.empty():
            '''
            print "ERROR: No data in the queue...Returning empty rather than exiting..."
            data = n.array([n.zeros(self.shape, dtype=float) for i in range(0,self.batch_size)])
            print "Done making empties"
            '''
            #or we could wait, assuming camera is not broken and accumulate images then...
            while self.batchQueue.qsize()<self.batch_size and self.do_not_exit and self.streaming:
                time.sleep(1)
                print "Waiting for camera to catch up...",self.batch_size,',',self.batchQueue.qsize()
            data = n.array([self.batchQueue.get() for i in range(0, self.batch_size)])
        else:
            data = n.array([self.batchQueue.get() for i in range(0, self.batch_size)])
        labels = n.array([0 for i in range(0,self.batch_size)])#check this value, testing?
        #return data
        return (data, labels) #do we need labels? how does it handle prediction?

    '''
    Adds duplicates without changing order:
    '''
    def stretch_to_other(self, l, n):
        #add full list copies
        comb = [l for i in range(0, n/len(l))]
        #add remainder
        #remains = l[0:n%len(l)]
        comb.append(l[0:n%len(l)])
        out = []
        #merge
        for idx in range(0, len(comb[0])):
            for item in comb:
                if idx < len(item):
                    out.append(item[idx])
        return out

    '''
    listener:
        gets a frame and adds it to the Queue
        meant to be used in a thread/called frequently while training/testing is happening
    '''
    def listener_standard(self):
        while (self.do_not_exit):
            #this happens if we are in testing
            #wait for frames
            openni2.wait_for_any_stream([self.depth_stream])
            openni2.wait_for_any_stream([self.color_stream])
            #grab a frame from each
            frame_color = self.color_stream.read_frame()
            frame_depth = self.depth_stream.read_frame()
            #change format and grab the data
            #frame_data_depth = frame_depth.get_buffer_as_uint16()
            #frame_data_color = frame_color.get_buffer_as_uint8()
            frame_data_depth = frame_depth.get_buffer_as(ctypes.c_uint16)
            frame_data_color = frame_color.get_buffer_as(ctypes.c_uint8)
            #cropping? this seems to be enabled and not enabled: returns true, but values not set...
            #add frame to Queue (for learning)
            #stow as numpy data
            depthData  = (n.frombuffer(frame_data_depth, dtype=n.uint16)).astype(dtype=n.single, order='C')
            #otherDepth = (n.frombuffer(frame_data_depth, dtype=n.uint16)).astype(dtype=n.uint16, order='C')
            colorData  = (n.frombuffer(frame_data_color, dtype=n.uint8)).astype(dtype=n.single, order='C')
            #reshape data
            depthData = CameraStreamer.reshape(depthData)
            #otherDepthData = CameraStreamer.reshape(otherDepth)
            colorData = (CameraStreamer.reshape(colorData))[:,:,::-1]
            #stack to form main image
            total = n.dstack((colorData, depthData))
            #add to batchQueue
            self.batchQueue.put(total)
            #keep queue size down
            if (self.batchQueue.qsize()>self.max_batch_size):
                #print "Dropping frame"
                #NOTE: if we want to not lose some of this data, we will want to 
                #decide which frame to remove more intelligently
                #ie: track what frames we're tossing, and remove every other (but it is a Queue...)
                #OR: automatically lower rate at which we accumulate them: 
                #change sleep time
                self.batchQueue.get()
            if self.pause_time>0:
                #print "Pausing"
                #add a pause between loops? Smaller? Larger? Not needed?
                time.sleep(self.pause_time)

    def listener_fast(self):
        while (self.do_not_exit):
            #print "Looping..."
            #this happens if we are in testing
            #wait for frames
            openni2.wait_for_any_stream([self.depth_stream])
            openni2.wait_for_any_stream([self.color_stream])
            #grab a frame from each
            frame_depth = self.depth_stream.read_frame()
            frame_color = self.color_stream.read_frame()
            #change format and grab the data
            frame_data_depth = frame_depth.get_buffer_as_uint16()
            frame_data_color = frame_color.get_buffer_as_uint8()
            #add frame to Queue (for learning)
            #stow as numpy data
            depthData  = (n.frombuffer(frame_data_depth, dtype=n.uint16)).astype(dtype=n.single, order='C')
            colorData  = (n.frombuffer(frame_data_color, dtype=n.uint8)).astype(dtype=n.single, order='C')
            #skip reshaping and stacking: let caller do that
            #add to batchQueue
            self.batchQueue.put((colorData, depthData))
            #keep queue size down
            if (self.batchQueue.qsize()>self.max_batch_size):
                #NOTE: if we want to not lose some of this data, we will want to 
                #decide which frame to remove more intelligently
                #ie: track what frames we're tossing, and remove every other (but it is a Queue...)
                #OR: automatically lower rate at which we accumulate them: 
                #change sleep time
                self.batchQueue.get()
            if self.pause_time>0:
                #add a pause between loops? Smaller? Larger? Not needed?
                time.sleep(self.pause_time)

    '''
    listener:
        gets a frame and adds it to the Queue
        meant to be used in a thread/called frequently while training/testing is happening
    '''
    def listener_standard_color(self):
        while (self.do_not_exit):
            #this happens if we are in testing
            #wait for frames
            openni2.wait_for_any_stream([self.color_stream])
            #grab a frame from each
            frame_color = self.color_stream.read_frame()
            #change format and grab the data
            #frame_data_color = frame_color.get_buffer_as_uint8()
            frame_data_color = frame_color.get_buffer_as(ctypes.c_uint8)
            #cropping? this seems to be enabled and not enabled: returns true, but values not set...
            #add frame to Queue (for learning)
            colorData  = (n.frombuffer(frame_data_color, dtype=n.uint8)).astype(dtype=n.single, order='C')
            #reshape data
            colorData = (CameraStreamer.reshape(colorData))[:,:,::-1]
            #add to batchQueue
            self.batchQueue.put(colorData)
            #keep queue size down
            if (self.batchQueue.qsize()>self.max_batch_size):
                #print "Dropping frame"
                #NOTE: if we want to not lose some of this data, we will want to 
                #decide which frame to remove more intelligently
                #ie: track what frames we're tossing, and remove every other (but it is a Queue...)
                #OR: automatically lower rate at which we accumulate them: 
                #change sleep time
                self.batchQueue.get()
            if self.pause_time>0:
                #print "Pausing"
                #add a pause between loops? Smaller? Larger? Not needed?
                time.sleep(self.pause_time)

    def listener_fast_color(self):
        while (self.do_not_exit):
            #print "Looping..."
            #this happens if we are in testing
            #wait for frames
            openni2.wait_for_any_stream([self.color_stream])
            #grab a frame from each
            frame_color = self.color_stream.read_frame()
            #change format and grab the data
            frame_data_color = frame_color.get_buffer_as_uint8()
            #add frame to Queue (for learning)
            #stow as numpy data
            colorData  = (n.frombuffer(frame_data_color, dtype=n.uint8)).astype(dtype=n.single, order='C')
            #skip reshaping and stacking: let caller do that
            #add to batchQueue
            self.batchQueue.put(colorData)
            #keep queue size down
            if (self.batchQueue.qsize()>self.max_batch_size):
                #NOTE: if we want to not lose some of this data, we will want to 
                #decide which frame to remove more intelligently
                #ie: track what frames we're tossing, and remove every other (but it is a Queue...)
                #OR: automatically lower rate at which we accumulate them: 
                #change sleep time
                self.batchQueue.get()
            if self.pause_time>0:
                #add a pause between loops? Smaller? Larger? Not needed?
                time.sleep(self.pause_time)

    '''
    Version: same as above except I am moving some processing from the dataprovider 
    to the camera streamer in an attempt to speed thigns up. Bad for modularity, 
    but I want to see if this helps.
    '''
    def listener_standard_patch(self):
        while (self.do_not_exit):
            #this happens if we are in testing
            #wait for frames
            openni2.wait_for_any_stream([self.depth_stream])
            openni2.wait_for_any_stream([self.color_stream])
            #grab a frame from each
            frame_color = self.color_stream.read_frame()
            frame_depth = self.depth_stream.read_frame()
            #change format and grab the data
            #frame_data_depth = frame_depth.get_buffer_as_uint16()
            #frame_data_color = frame_color.get_buffer_as_uint8()
            frame_data_depth = frame_depth.get_buffer_as(ctypes.c_uint16)
            frame_data_color = frame_color.get_buffer_as(ctypes.c_uint8)
            #cropping? this seems to be enabled and not enabled: returns true, but values not set...
            #add frame to Queue (for learning)
            #stow as numpy data
            depthData  = (n.frombuffer(frame_data_depth, dtype=n.uint16)).astype(dtype=n.single, order='C')
            #otherDepth = (n.frombuffer(frame_data_depth, dtype=n.uint16)).astype(dtype=n.uint16, order='C')
            colorData  = (n.frombuffer(frame_data_color, dtype=n.uint8)).astype(dtype=n.single, order='C')
            #reshape data
            depthData = CameraStreamer.reshape(depthData)
            #otherDepthData = CameraStreamer.reshape(otherDepth)
            colorData = (CameraStreamer.reshape(colorData))[:,:,::-1]
            #stack to form main image
            total = n.dstack((colorData, depthData))
            #Move functionality from get_batch_from_camera in data provider here...

            #scale depth if being used down to same scale as rest of data

            #subtract total mean if set #requires access to total mean image

            #grab patches #requires access to resolutions used to make patches

            #subtract mean patch #requires mean patch

            #If we want to move anymore than this (ie, reshaping) we'll need to alter get_next_batch as well...perhaps we should.

            #add to batchQueue
            self.batchQueue.put(total)
            #keep queue size down
            if (self.batchQueue.qsize()>self.max_batch_size):
                #print "Dropping frame"
                #NOTE: if we want to not lose some of this data, we will want to 
                #decide which frame to remove more intelligently
                #ie: track what frames we're tossing, and remove every other (but it is a Queue...)
                #OR: automatically lower rate at which we accumulate them: 
                #change sleep time
                self.batchQueue.get()
            if self.pause_time>0:
                #print "Pausing"
                #add a pause between loops? Smaller? Larger? Not needed?
                time.sleep(self.pause_time)

    '''
    begin_streaming:
        Now we want camera frames to be accumulated:
            start thread that uses listener_callback above.
            rejoin in end_streaming below
    '''    
    def begin_streaming(self, fast=False):
        #begin both streams
        if self.both_streams:
            self.depth_stream.start()
        self.color_stream.start()
        time.sleep(0.2)#wait for camera to settle (adjust to lighting)
        self.streaming = True
        self.do_not_exit = True
        #now what we really need is callbacks/threads...
        #begin a separate thread for the camera stream:
        #accumulates frames in a numpy data structure until full, 
        #then begins to drop old for new... would be better if time assurances
        #Only works as long as Classifier 'faster' then video...syncing thoughts?
        if self.both_streams:
            if fast:
                print "starting fast thread"
                self.t = threading.Thread(target=self.listener_fast, args=()) #do we need this self?
            else:
                print "starting normal thread"
                self.t = threading.Thread(target=self.listener_standard, args=())
        else:
            #color only threads
            if fast:
                print "starting fast thread"
                self.t = threading.Thread(target=self.listener_fast_color, args=()) #do we need this self?
            else:
                print "starting normal thread"
                self.t = threading.Thread(target=self.listener_standard_color, args=())
        self.t.daemon = True #kill it if main program ends
        self.t.start() #should begin the function: loop until stopped
        '''
        at this point we switch timing methods: it would be great 
        if, instead of the trainer having control (give me the next 
        batch now), another process took over:
            *If we are falling behind: begin new test batchnum
            *If we have nothing running: begin a new batchnum
            *otherwise accumulate stream for next batch
        '''

    '''
    stop the camera and close openni
    '''
    def end_streaming(self):
        #end the streaming thread
        print "Trying to end thread"
        self.do_not_exit = False
        #wait for thread to get the message: bad if this fails
        self.t.join()
        print "Thread ended"
        #stop streams
        if self.both_streams:
            self.depth_stream.stop()
        self.color_stream.stop()

    def clear_queue(self):
        value = self.batchQueue.qsize()
        while not self.batchQueue.empty():
            self.batchQueue.get()
        return value

    def close_camera(self):
        #ditch camera
        openni2.unload()#and nite.unload()?
        print "closed openni, stopped depth streams"

    def display_depth(self, depth):
        #depth is width x height x 1
        #scale values
        i,j,k = np.unravel_index(depth.argmax(), depth.shape)
        a,b,c = np.unravel_index(depth.argmin(), depth.shape)
        image = np.copy((depth.astype(np.single)*(255.0/depth[i,j,k]))).astype(np.uint16)
        cv2.imshow('Depth',image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''plt.imshow(temp.astype(numpy.uint8))
        plt.pause(.1)
        plt.draw()'''

    def display_color(self, color):
        #depth is width x height x 1
        #scale values
        image = np.copy(color).astype(np.uint8)
        cv2.imshow('Depth',image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''plt.imshow(temp.astype(numpy.uint8))
        plt.pause(.1)
        plt.draw()'''

    @classmethod
    def reshape(cls, img):
        assert len(img.shape) == 1, "Image Shape is Multi-Dimensional before reshaping "+str(img.shape)
        whatisit = img.shape[0]
        #print "Initial Size in reshape ",whatisit
        if whatisit == (320*240*1): #QVGA, not sure how to change this yet...
            img.shape = (1, 240, 320)#small chance these may be reversed in certain apis...
            #This order? Really? ^
            #img = n.concatenate((img, img, img), axis=0)
            img = n.swapaxes(img, 0, 2)
            img = n.swapaxes(img, 0, 1)
        elif whatisit == (320*240*3):
            img.shape = (240, 320, 3)
            #these are, what is it, normalizsed?
        elif whatisit == (640*480*1):
            img.shape = (1, 480, 640)#small chance these may be reversed in certain apis...
            #This order? Really? ^
            #img = n.concatenate((img, img, img), axis=0)
            img = n.swapaxes(img, 0, 2)
            img = n.swapaxes(img, 0, 1)
        elif whatisit == (640*480*3):
            img.shape = (480, 640, 3)
        else:
            print "Frames do not match any image format. Frames are of size: ",img.size
            sys.exit()
        #print "Final size in reshape ",img.shape
        return img

    '''
    Prints the debugging data: device options
    '''
    def print_video_options(self, device, streamType):
        stream_info = self.device.get_sensor_info(streamType)
        print "Stream format options: ", streamType
        for item in stream_info.videoModes:
            print item

    '''
    Prints the frame using matplot lib (slow)
    '''
    def print_frame(self, frame_data, thisType):
        img  = n.frombuffer(frame_data, dtype=thisType)
        whatisit = img.size
        print "Image size ",whatisit
        if whatisit == (320*240*1): #QVGA, not sure how to change this yet...
            img.shape = (1, 240, 320)#small chance these may be reversed in certain apis...
            #This order? Really? ^
            img = n.concatenate((img, img, img), axis=0)
            img = n.swapaxes(img, 0, 2)
            img = n.swapaxes(img, 0, 1)
        elif whatisit == (320*240*3):
            img.shape = (240, 320, 3)
            #these are, what is it, normalizsed?
        elif whatisit == (640*480*1):
            img.shape = (1, 480, 640)#small chance these may be reversed in certain apis...
            #This order? Really? ^
            img = n.concatenate((img, img, img), axis=0)
            img = n.swapaxes(img, 0, 2)
            img = n.swapaxes(img, 0, 1)
        elif whatisit == (640*480*3):
            img.shape = (480, 640,3)
        else:
            print "Frames do not match any image format. Frames are of size: ",img.size
            sys.exit()
        #shape it accordingly
        print img.shape
        '''plt.imshow(img)
        plt.show()'''
        cv2.imshow('Depth',image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

'''
Test code: debugging
'''
def testStreamer():
    print "Beginning Streaming"
    batch_size = 2
    streamer = CameraStreamer(batchSize=batch_size, size = 'quarter', max_lag=2, max_size=40)
    streamer.begin_streaming(False)

    #wait for some images to accumulate
    time.sleep(10)

    streamer.end_streaming()
    batch = streamer.get_next_batch()
    streamer.close_camera()

if __name__ == "__main__":
    testStreamer()
