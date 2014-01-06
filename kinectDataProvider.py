'''
Temporary KinectData Provider, for testing before adding to the other file of data providers.

'''

'''
There are a few other constraints on the data provider:

    *The data provider must define a function get_data_dims(self, idx=0), which returns the data 
dimensionality of each matrix returned by get_next_batch. 
    *The data matrix must be C-ordered (this is the default for numpy arrays), with dimensions 
(data dimensionality)x(number of cases). If your images have channels (for example, the 3 
 color channels in the CIFAR-10), then all the values for channel n must precede all the 
values for channel n + 1.
    *The data matrix must consist of single-precision floats.

Assuming you're going to use the logistic regression objective built into the code rather than 
write your own, there are a few additional constraints on the data provider:

    *The labels vector (returned by get_next_batch) must have dimensionality 1x(number of cases).
    *The elements of the labels vector must range from 0 until the number of classes minus one.
    *The labels vector must consist of single-precision floats.
    *The data provider must define a function get_num_classes(self) which simply returns the number 
of classes in your dataset. The CIFARDataProvider inherits this function from LabeledDataProvider 
defined in data.py.

Fill batch with smaller images: 64x64 etc.
remove softmax layer for mutliclass labeling:
    Also have to write our own logisitc regression objective since theirs assumes labels 1 x numCases
add 'unknown' class to dictionary: need total dictionary across images: store in batches.meta
So batch per frame : split picture into smaller little pictures

NOTE: Try resizing all patches: network gets nans on 'large' input?
'''

import sys, time
#from primesense import nite
from data import *
import numpy.random as nr
import numpy as n
import random as r
import itertools #flattens
#temporary for checking display:
#import cv2
#cv2 removed due to conflicts with matplotlib gtk versions
import matplotlib.pyplot as pl
from cameraStreamer import *

from util import *

def display_depth(depth):
    #depth is width x height x 1
    #scale values
    i,j,k = n.unravel_index(depth.argmax(), depth.shape)
    a,b,c = n.unravel_index(depth.argmin(), depth.shape)
    image = (n.copy(depth)*(255.0/depth[i,j,k])).astype(n.uint8)
    #cv2.imshow('Color',image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    pl.imshow(image)
    pl.show()

def display_color(image):
    #show image
    #cv2.imshow('Color',image.astype(n.uint8))
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    pl.imshow(image)
    pl.show()

def emergency_quit():
    s = raw_input('--> ')
    if s=='q':
        sys.exit()

def emergency_print(image):
    print "Full Image, all channels"
    cv2.imshow('Color',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print "Full Image, first Channel"
    cv2.imshow('Color',image[:,:,0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    emergency_quit()
    print "Full Image, second channel"
    cv2.imshow('Color',image[:,:,1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print "Full Image, third channel"
    cv2.imshow('Color',image[:,:,2])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print "Full Image, fourth channel"
    cv2.imshow('Color',image[:,:,3])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    emergency_quit()

def widenCrop(mini, crop, resolution):
    assert resolution[0]>mini and resolution[1]>mini,"Cannot widen crop any further: image size smaller than needed resolution."
    xDiff = (crop[2]-crop[0])
    yDiff = (crop[3]-crop[1])
    while(xDiff<mini or yDiff<mini):
        xDiff = max(mini-xDiff, 0)
        yDiff = max(mini-yDiff, 0)
        crop0 = max(crop[0] - (xDiff/2 + 1), 0) #add half to this side, half to other side
        crop1 = max(crop[1] - (yDiff/2 + 1), 0)
        crop2 = min(crop[2] + (xDiff/2 + 1), resolution[0]-1)
        crop3 = min(crop[3] + (yDiff/2 + 1), resolution[1]-1)
        crop = (crop0, crop1, crop2, crop3)
        xDiff = (crop[2]-crop[0])
        yDiff = (crop[3]-crop[1])
    assert crop[0]<crop[2] and crop[1]<crop[3], "Wider crop is incorrect "+str(crop[0])+"<"+str(crop[2])+"  "++str(crop[1])+"<"+str(crop[3])
    return crop

class KinectDataProvider(LabeledMemoryDataProvider):#Labeled or LabeledMemory
    #still loads training from file, but grabs from Kinect for testing (or from file, both options)
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        LabeledDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)

        self.data_dic=[] #load in get_next_batch (contains dictionaries of data and labels (and batch_label and filenames))
        #do we grab tests from camera or from file
        print "Parameters: "
        for item in dp_params.keys():
            print item

        #to make up for previous versions, where these options did not yet exist: (backwards comp...kinda sucks)
        req_ops = {'test_from_camera': False, 'multi_label': False, 'resolutions': [96], 'with_depth': False, 
            'full_resolution': True, 'subtract_mean_patch': False, 'use_drop_out': True, 'scale_depth': False}
        #check for existance of each, else default
        for item in req_ops.keys():
            if item in dp_params:
                setattr(self, item, dp_params[item])
            else:
                setattr(self, item, req_ops[item])

        #the following should be set in dp_params eventually, for now set it here
        #self.test_from_camera = dp_params['test_from_camera']#False 
        #self.multilabel = dp_params['multi_label']#False #select single or many labels allowed
        #self.flatlabels = True #replaced with sensible type check.
        #self.multiresolution = dp_params['resolutions']#False #replace with resolution list: if it is non-empty this is true
        #self.resolutions = dp_params['resolutions']
        #self.with_depth = dp_params['with_depth']#False
        #self.full_resolution = dp_params['full_resolution']#True #on certain machines larger resolutions cannot be read from camera
        #self.subtract_mean_patch = dp_params['subtract_mean_patch']#True #helpful only if single resolution, no resizing done...
        #self.use_drop_out = dp_params['use_drop_out']
        #self.scale_depth = dp_params['scale_depth']
        self.word = "" #we still have a whole dictionary, and aren't doing multilabel yet.

        #TESTING OPTIONS: (These may end up being always True or False, not sure if they matter yet)
        self.crop_general = False        #probably a bad idea, always False
        self.with_none_class = True     #probably a good idea, needs further testing
        self.max_batch = 14000           #may be uneccessary but a good idea for consistent timing...
        print "Output size",len(self.batch_meta['label_names'])

        #add NONE label if it does not exist
        if self.with_none_class and not 'NONE' in self.batch_meta['label_names']:
            self.batch_meta['label_names'].append('NONE')

        #print "LABELS: "
        #for item in self.batch_meta['label_names']:
        #    print item

        self.batchSize = 30#48 # hard coded from data for now
        self.scale = 1.0 #0.5 # hard code scaling of the images...
        #self.final_size = 4
        #storage for multiple resolution selections
        if len(self.resolutions) == 0:
            self.resolutions.append(96)
        if (self.full_resolution):
            self.img_size = 480 #Maximum for square
        else:
            self.img_size = 240 #or what?
        self.inner_size = self.resolutions[0]#96 #64 #make images 64 by 64
        self.final_size = dp_params['final_res']#48 #images need to be much smaller?

        #set up to have same size images: should force the same size batch for each training image
        if self.crop_general:
            self.overall_crop = self.batch_meta['overall_crop']
        else:
            self.overall_crop = None

        self.test = test
        if (self.full_resolution):
            #using 480, 640, 4
            self.full_img_size = self.batch_meta['num_vis'] #include depth for now
            #cutting out depth
            if not self.with_depth:
                self.full_img_size = (self.full_img_size[0], self.full_img_size[1], self.full_img_size[2]-1)
            #print "Image shape ", self.full_img_size
        else:
            #using quarter size
            assert len(self.batch_meta['num_vis']) == 3, "Image shape wrong "+str(self.batch_meta['num_vis'])
            self.full_img_size = (self.batch_meta['num_vis'][0]/2.0, self.batch_meta['num_vis'][1]/2.0,self.batch_meta['num_vis'][2])
            #cutting out depth
            if not self.with_depth:
                self.full_img_size[2] = (self.full_img_size[0], self.full_img_size[1], self.full_img_size[2]-1)
        #Can we be sure labeled thing is in all of these? depends entirely on size, postion etc...
        if self.with_depth:
            self.num_colors = 4 #assume depth here? read from training?
        else:
            self.num_colors = 3
        if self.test and self.test_from_camera:
            if self.full_resolution:
                self.streamer = CameraStreamer(self.batchSize, size='full', with_depth=self.with_depth, max_size=self.batchSize)#batch_size gotten from batches.meta?
            else:
                self.streamer = CameraStreamer(self.batchSize, size='quarter', with_depth=self.with_depth, max_size=self.batchSize)

        #why are the means ordered this way in cifar? channels then shape?
        #mean is very troubling: want patches, but need to be able to - and add mean...
        if self.with_depth:
            self.data_mean = self.batch_meta['data_mean']#.reshape((self.full_img_size))
            if self.scale_depth:
                #scale depth
                minVal = 0.0
                maxVal = 9870.0
                self.data_mean[:,:,3] = n.round(255.0 * (self.data_mean[:,:,3] - minVal) / (maxVal - minVal - 1.0))
        else:
            self.data_mean = self.batch_meta['data_mean'][:,:,:3]
        #deconstruct mean for adding back in? tedious patches...
        patches, locs = self.multi_res_patches(self.data_mean)
        self.mean_patch = n.mean((patches), axis=0)#across images (each pixel channel has mean)
        assert self.mean_patch.shape == (self.final_size, self.final_size, self.num_colors), "Mean patch is wrong shape "+str(self.mean_patch.shape)

        #TODO replace self.batch_meta['labels'] with something reflecting the single label choice if needed for multiclass allowed

    def scale_data(self, data):
        #Question: scale across images, across pixels, across all...
        #a = a / float(max(a)) * new_max
        new_max = 1.0
        old_max = float(data.max()) #data[0].max()#image #data.max(axis=0) #pixels...

    def find_consistent_crop(self):
        #cropBoxes = batch['cropboxes']
        cropBoxes=[]#problem accessing this across batches
        #find largest cropping and use it
        maxPoint = [0,0]
        minPoint = [900,900]#asume images smaller than this
        for item in cropBoxes:
            #fixCrop[i][2]:fixCrop[i][0], fixCrop[i][3]:fixCrop[i][1]
            if item[0]> maxPoint[0]:
                maxPoint[0] = item[0]
            if item[1]>maxPoint[1]:
                maxPoint[1] = item[1]
            if item[2]<minPoint[0]:
                minPoint[0] = item[2]
            if item[3]<minPoint[1]:
                minPoint[1] = item[3]
        self.overall_crop = (maxPoint[0],maxPoint[1], minPoint[0], minPoint[1])

    '''
    After some profiling tests, this somehow proved faster than several other methods, see Drop Test ipython notebook
    '''
    def drop_out(self, data, percent=.8):
        #for each image in data (diff 20% for each image in batch)
        for i in range(0,len(data)):
            #create array same size as image
            #randomly have 0 and 1 in it (20% zeros)
            dropper = ((n.random.binomial(1, percent, (data[0].shape[0]*data[0].shape[1]))).astype(n.single)).reshape((data[0].shape[0],data[0].shape[1],1))
            dropped=dropper
            for j in range(1, data[0].shape[-1]):
                dropper = n.concatenate((dropped,dropper), axis=2)
            #multiply elementwise
            data[i][:,:,:] = n.multiply(dropper, data[i][:,:,:])
            #would it be faster to do this all at once or per image? Let's find out
        #return data with blanks 
        return data #?
        #create separate random mask for each sample and apply mask to each sample
        #return numpy.array([numpy.ma.array(item,mask=numpy.random.binomial(1,percent,data[0].shape),fill_value=0.0) for item in data])

    def get_next_batch(self, test_random=False):
        #also needs to return epoch and batchnum...ok
        epoch, batchnum = self.curr_epoch, self.curr_batchnum
        self.advance_batch()
        #get data from camera
        if test_random:
            (data, labels, locs), images = self.get_batch_from_file_random()
        elif self.test and self.test_from_camera: #make sure this is correct: when does swap from train to predict
            (data, labels, locs), images = self.get_batch_from_camera()
        #otherwise load from file...?
        else:
            #get_next_batch
            (data, labels, locs), images = self.get_batch_from_file()
        #return the data for running through network, (and label matrix too)
        #print " \nType and shape ",type(data),' ',len(data),' ',type(data[0]), data[0].shape, ',',data[0].dtype
        startTime = time.time()
        if len(data) > self.max_batch:
            #print "how many samples: ",len(data),',',self.max_batch
            data = data[:self.max_batch]
            labels = labels[:self.max_batch]
            locs = locs[:self.max_batch]
        print "Cutting batch ",time.time()-startTime
        #dropout: 20% pixels zeroed?
        startTime = time.time()
        if self.use_drop_out:
            data = self.drop_out(data)
        print "Drop Out ",time.time()-startTime
        #print " \nType and shape ",type(data),' ',len(data),' ',type(data[0]), data[0].shape, ',',data[0].dtype
        imageTest = n.copy(data[0])
        #flattened and each channel in order...
        startTime = time.time()
        data = n.asarray([(sample.astype(n.float32, order='C').swapaxes(0,1)).flatten('F') for sample in data])#flatten data, keep channels in order
        labels = n.asarray(labels)#[label for label in labels]
        print "Reshaping data 1 ",time.time()-startTime
        #assert data.shape[1] == (self.final_size * self.final_size * self.num_colors), "Data for next batch is not the correct shape "+str(data.shape)
        #print "Batch dims: ",data.shape,',',labels.shape
        #need to flatten the data and swap axis order: data x numSamples
        startTime = time.time()
        data = n.require(n.swapaxes(data, 0, 1), dtype=n.single, requirements='C')#n.swapaxes(data.astype(n.float32, order='C'), 0, 1)
        labels = n.require(n.swapaxes(labels.reshape(labels.shape[0],1), 0, 1), dtype=n.single, requirements='C')#n.swapaxes(labels.astype(n.float32, order='C').reshape(labels.shape[0],1), 0, 1)
        print "Reshaping data 2 ",time.time()-startTime
        #print " \nType and shape ",type(data),' ',data.shape,' ',type(data[0]), data[0].shape, ',',data[0].dtype
        #print "Fixed Batch dims: ",data.shape,',',labels.shape
        #print imageTest[:,:,0]
        #print "blah"
        #print data[:4*4,0]
        #assert round(imageTest[0,0,0]-data[0,0], 4)==0, "Channels swapped order? "+str(imageTest[0,0,0])+' '+str(imageTest[1,0,0])+' '+str(imageTest[0,1,0])+' '+str(imageTest[0,0,1])+','+str(data[0][0])+' '+str(data[1][0])
        #assert round(imageTest[0,1,0]-data[1,0], 4)==0, "Channels swapped order? "+str(imageTest[0,1,0])+','+str(data[1,0])
        #print type(data[0]),',',type(labels[0])
        #print data[0].shape,',',labels[0].shape
        print "Handing off data "
        return epoch, batchnum, [data, labels], locs, images ###recon flag? [0]==epoch, [1]==batchnum, [2]==[data,labels], [3]==locs

    def get_data_dims(self, idx=0):
        #this does not include the batch size, just image/sample
        #well, 640x480x3 or 4, then X number of samples per batch...
        if self.multilabel:
            #print self.inner_size**2 * self.num_colors
            return self.final_size**2 * self.num_colors if idx == 0 else self.get_num_classes
        else:
            #print self.inner_size**2 * self.num_colors
            return self.final_size**2 * self.num_colors if idx == 0 else 1

    #inherited version works : only if doing multiple classes
    def get_num_classes(self):
        #return 2: if self.word !="" (0, 1) is or is not in class self.word
        if self.word != "":
            return 2
        else:
            return len(self.batch_meta['label_names'])

    def get_batch_from_camera(self):
        #start streamer if needed
        if self.streamer.streaming==False:
            self.streamer.begin_streaming()
        #get the batch
        (batchData, batchLabels) = self.streamer.get_next_batch()
        #display_color(batchData[1][:,:,:3][:,:,::-1].astype(n.uint8))
        #scale depth down to similar range as rest of data...
        #so: # no done before mean subtraction, scale depth portion of mean too
        startTime = time.time()
        if self.scale_depth:
            #scale the last dimension of every sample to be 0-255
            minVal = 0.0
            maxVal = batchData[:,:,:,3].max()#max across all depth: this is temp est
            #assert maxVal==9870, "Max changed "+str(maxVal)+','+str(9870)
            if maxVal!=9870:
                print "Max value varies ",maxVal,',',9870
            #scaled_depth = np.round(255.0 * (depth - minVal) / (maxVal - minVal - 1.0))
            #batchData[:,:,:,3] = n.round(255.0 * (batchData[:,:,:,3] - minVal) / (maxVal - minVal - 1.0))
            batchData[:,:,:,3] = n.round(255.0 * (batchData[:,:,:,3] - minVal) / (9870.0 - minVal - 1.0))
        print "scaling depth: ",time.time()-startTime
        #have full images, subtract mean
        startTime= time.time()
        if not self.subtract_mean_patch:
            fixed_batch_data  = [image-self.data_mean for image in batchData]
        else:
            fixed_batch_data = batchData
        print "remove mean total: ",time.time()-startTime
        #now: do any cropping and scaling
        #grab self.inner_size x self.inner_size throughout images
        cropped_batch_data = []
        cropped_batch_labels = []
        cropped_batch_locations = []
        startTime = time.time()
        for i,image in enumerate(fixed_batch_data):
            #grab patches several resolutions resized to be same
            patches, locations = self.multi_res_patches(image)
            locations = [item+(i,) for item in locations] 
            cropped_batch_data = cropped_batch_data + patches
            cropped_batch_locations = cropped_batch_locations + locations
            #extend label by length of patches (label for image applies to each patch from image (Maybe))
            cropped_batch_labels = cropped_batch_labels + [batchLabels[i] for j in range(0, len(patches))]
        print "making patches: ",time.time()-startTime
        #many more now throughtout image...but: debug check here
        #assert len(cropped_batch_data) == len(cropped_batch_labels), \
        #    "Not the same number of samples and labels "+str(len(cropped_batch_data))+','+str(len(cropped_batch_labels))
        startTime = time.time()
        if self.subtract_mean_patch:
            #subtract patch rather than whole image mean:
            #assert type(cropped_batch_data[0]) == type(self.mean_patch), "Mean patch and image patch have different types "+str(type(cropped_batch_data[0]))+","+str(type(self.mean_patch))
            #assert cropped_batch_data[0].shape == self.mean_patch.shape, "Mean patch is different shape than the image patches "+str(self.mean_patch.shape)+","+str(cropped_batch_data[0].shape)
            cropped_batch_data = [item-self.mean_patch for item in cropped_batch_data]
        print "remove mean patch: ",time.time()-startTime
        print "returning camera batch"
        #just return
        return (cropped_batch_data, cropped_batch_labels, cropped_batch_locations), batchData

    def multi_res_patches(self, image):
        cropped_batch = []
        cropped_locations = []
        for resolution in self.resolutions:
            patches, locations = self.getPatches(image, resolution, int(resolution/2), self.final_size)
            #print type(what),',',len(what)
            cropped_batch = cropped_batch + patches
            cropped_locations = cropped_locations + locations
        return cropped_batch, cropped_locations

    def getPatches(self, image, resolution=64, stride=-1, finalRes=64):
        #print "getting patches of size ",resolution," from image ",image.shape
        #no patches here
        if image.shape[0]<resolution or image.shape[1]<resolution:
            return [], []
        #check stride, reset if invalid
        if stride == -1:
            stride = int(resolution/2)
        #storage for result patches
        patches = []
        #store (upperleftx, upperlefty, size)
        locations = []
        #keeping square patches only
        width=height=resolution
        #iterate through rows
        for i in range(0,  image.shape[0]-(image.shape[0]%width)+1, stride):#from 0 to right #-width instead?
            #iterate through columns
            for j in range(0, image.shape[1]-(image.shape[1]%height)+1, stride):#from 0 to bottom #-height instead?
                #check if we went over the x & y image boundary; Fill with 0 if we did
                if (i+width>image.shape[0] and j+height>image.shape[1]):
                    #slide back over
                    patch = image[image.shape[0]-width:image.shape[0], image.shape[1]-height:image.shape[1]]
                    location = (image.shape[0]-width, image.shape[1]-height, resolution)
                #check if we went over the x image boundary
                elif (i+width>image.shape[0]):
                    #slide back over
                    patch = image[image.shape[0]-width:image.shape[0], j:j+height]
                    location = (image.shape[0]-width, j, resolution)
                #check if we went over the y image boundary
                elif (j+height>image.shape[1]):
                    #slide back over
                    patch = image[i:i+width, image.shape[1]-height:image.shape[1]]
                    location = (i,image.shape[1]-height, resolution)
                #fine: within image, just get patches
                else:
                    patch = image[i:i+width, j:j+height]
                    location = (i,j,resolution)
                #check that 
                if resolution != finalRes:
                    #resize our patch to be correct scale
                    #patch = misc.imresize(patch, (finalRes, finalRes), interp='bilinear', mode=None)
                    patch = cv2.resize(patch, (finalRes, finalRes))
                patches.append(patch)
                locations.append(location)
        # add return of locations list
        #print "debug ",len(patches),',',len(locations)
        return patches, locations #,[(upperleftx, upperlefty, size),...]

    def get_patches_about_crop(self, image, crop=(), resolution=64, stride=-1):
        if crop == ():
            #no cropping: return patches everywhere IN, empty list for Patches OUT
            patchesIn, patchInLocs = self.multi_res_patches(image, resolution, stride)
            return patchesIn, [], patchesInLoc, []
        patchesIn, patchesInLoc = self.multi_res_patches(image[crop[0]:crop[2],crop[1]:crop[3],:])
        if self.with_none_class:
            patchesOut, patchesOutLoc = zip(
                                            list(self.multi_res_patches(image[0:crop[0],0:image.shape[1],:])),
                                            list(self.multi_res_patches(image[crop[0]:crop[2],0:crop[1],:])),
                                            list(self.multi_res_patches(image[crop[0]:crop[2],crop[3]:image.shape[1],:])),
                                            list(self.multi_res_patches(image[crop[2]:image.shape[0],0:image.shape[1],:]))
                                            )
            patchesOut = list(itertools.chain(*patchesOut))
            patchesOutLoc = list(itertools.chain(*patchesOutLoc))
            #DEBUG: remove these slow assertions later
            for item in  patchesOut:
                assert type(item) == type(patchesOut[0]), "Type wrong for some item in the outPut "+str(type(item))
            #print type(patchesOut[0]),',',patchesOut[0]
            for item in  patchesOutLoc:
                assert type(item) == type(patchesOutLoc[0]), "Type wrong for some item in the outPutLoc "+str(type(item))
        else:
            patchesOut = []
            patchesOutLoc = []
        #return patches and locs
        return patchesIn, patchesInLoc, patchesOut, patchesOutLoc

    '''
    Correctly sets up labels: multi word, single range, multiple...
    '''
    def get_labels_from_file(self, batch):
        labels = batch['labels']
        #print "\nDebug l ",len(labels)
        #print "DEBUG ",type(labels),',',type(labels[0]),',',self.multilabel
        if not self.multilabel and type(labels[0])=='dict':#not self.flatlabels:
            #print "not flat"
            #collect single value for chosen word
            listCountForWord = [labels[i][self.word] if self.word in labels[i] else -1 for i in range(0,len(labels))]
            #now to change counts to labels (0->classes-1)
            labelsMid = []
            for i in range(0, len(listCountForWord)):
                if listCountForWord[i]<=0:
                    labelsMid.append(0)#hasn't word
                else:
                    labelsMid.append(1)#has word
        if not self.multilabel and type(labels[0])!='dict': #self.flatlabels:
            #collect single value for chosen word
            #print "Flat"
            #for item in labels:
            #    print labels
            labelsMid = labels.astype(n.uint8)
        else:
            #make up default labels...? failed to find them exit
            print "Failed to find labels from file "
            sys.exit(0)
        #for label in labelsMid:
        #    print type(label),',',label.dtype,',',type(self.get_num_classes())
        #    print "What? ",label
        #    assert label<self.get_num_classes(), "ERROR: Label out of range "+str(label)+'<'+str(self.get_num_classes())
        return labelsMid

    def get_crop_from_file(self, batch):#, data):
        #if we're training, crop out things that are not the object...
        fixCrop = []
        #print "Cropping Training"
        if not self.overall_crop:
            cropBoxes = [widenCrop(self.inner_size, item, self.full_img_size) for item in batch['crop_boxes']]
        else:
            newCrop  = widenCrop(self.inner_size, self.overall_crop, self.full_img_size)
            cropBoxes = [newCrop for i in range(0, len(batch['crop_boxes']))]
        #dataCropped = [data[i][fixCrop[i][0]:fixCrop[i][2],fixCrop[i][1]:fixCrop[i][3],:] for i in range(0, len(fixCrop))]
        return cropBoxes

    def get_patched_data_from_file(self, data, labels, cropboxes):
        assert len(data) == len(labels), \
            "Length of data and labels not the same "+str(len(data))+" "+str(len(labels))
        assert len(data) == len(cropboxes), \
            "Length of data and cropboxes not the same "+str(len(data))+" "+str(len(cropboxes))
        #grab patches just from that area (at multiple resolutions?)
        finaldata = []
        finallabels = []
        finallocs = []
        imgidx=0
        #print "Collecting Patches"
        for image, label, cropbox in zip(data, labels, cropboxes):
            #get patches of image using crop box
            patchesIn, patchesInLoc, patchesOut, patchesOutLoc = self.get_patches_about_crop(image, cropbox)
            #remove some patchesOut so less background training:
            #shuffle so they're at random depths, and places in image
            #r.shuffle(patchesOut) #shuffle locations too...
            patchesOut_shuf = []
            patchesOutLoc_shuf = []
            index_shuf = range(len(patchesOut))
            r.shuffle(index_shuf)
            for i in index_shuf:
                patchesOut_shuf.append(patchesOut[i])
                patchesOutLoc_shuf.append(patchesOutLoc[i])
            patchesOut = patchesOut_shuf
            patchesOutLoc = patchesOutLoc_shuf
            #drop some...say force length same as the patchesIn for now
            if len(patchesIn)<len(patchesOut): #less test
                patchesOut = patchesOut[:len(patchesIn)/4]
                patchesOutLoc = patchesOutLoc[:len(patchesOut)]
            #add the patches
            finaldata = finaldata + patchesIn + patchesOut
            #multiply labels to be same as numebr of patchesIn
            finallabels = finallabels + [ label for i in range(0, len(patchesIn)) ]
            #ahhhh :D
            #finallocs = [(x,y,res), ...] #need: = [(x,y,res,imgidx), ...]
            patchesInLoc = [item+(imgidx,) for item in patchesInLoc] #the all important comma:tuple not int
            patchesOutLoc = [item+(imgidx,) for item in patchesOutLoc]
            #combine with existing
            finallocs = finallocs + patchesInLoc + patchesOutLoc
            #add NONE labels to be equal to number of patchesOut (background patches)
            noneLabel = self.batch_meta['label_names'].index('NONE')
            finallabels = finallabels + [ noneLabel for i in range(0, len(patchesOut)) ]
            #check final lengths
            assert len(finaldata) == len(finallabels)
            imgidx = imgidx + 1
        #subtract mean patch
        if self.subtract_mean_patch:
            assert finaldata[0].shape == self.mean_patch.shape, \
                "Mean patch is different shape than the image patches "+str(self.mean_patch.shape)+","+str(finaldata[0].shape)
            finaldata = [item-self.mean_patch for item in finaldata]
        return finaldata, finallabels, finallocs

    def bgr_to_rgb(self, image):
        return image[:,:,::-1]
        #return n.concatenate((image[:,:,2].reshape(image.shape[0], image.shape[1], 1),image[:,:,1].reshape(image.shape[0], image.shape[1], 1),image[:,:,0].reshape(image.shape[0], image.shape[1], 1)), axis=2)

    '''
    loads data and labels and formats them according to params
    '''
    def get_batch_from_file(self):
        #use self.word to select labels
        #others loaded all at start, here we load a batch from file at a time...
        batch = unpickle(self.get_data_file_name(self.curr_batchnum - self.batch_range[0]))
        #debug func
        #self.check_image_crop(batch)
        origData = batch['data']
        #labels = batch['labels']
        #assert len(origData) == len(labels), "ERROR: Not the same number of labels and data in batch"
        #display_color(origData[1][:,:,:3][:,:,::-1].astype(n.uint8))
        #need to parse thisBatch into proper format: pull out images and stack for training;
        #scale depth down to similar range as rest of data...
        #so: # no done before mean subtraction, scale depth portion of mean too
        if self.scale_depth:
            #scale the last dimension of every sample to be 0-255
            minVal = 0.0
            maxVal = origData[:,:,:,3].max() #max across all depth
            #assert maxVal==9870, "Max changed "+str(maxVal)+','+str(9870)
            if maxVal!=9870:
                print "Max value varies ",maxVal,',',9870
            #scaled_depth = np.round(255.0 * (depth - minVal) / (maxVal - minVal - 1.0))
            #origData[:,:,:,3] = n.round(255.0 * (origData[:,:,:,3] - minVal) / (maxVal - minVal - 1.0))
            origData[:,:,:,3] = n.round(255.0 * (origData[:,:,:,3] - minVal) / (9870.0 - minVal - 1.0))
        #pull out labels for self.word and stack for label vector 
        if not self.subtract_mean_patch:
            origData = origData-self.data_mean
        #resize if needed now
        if not self.full_resolution:
            origData = [misc.imresize(origData[i][0],(self.full_img_size[0],self.full_img_size[1]), interp='bilinear', mode=None) for i in range(0,len(origData))]
        #remove the depth if needed: after mean removed
        if not self.with_depth:
            #cut last channel of each item
            #fullData = [item[:,:,:3] for item in fullData]
            origData = origData[:,:,:,:3]
        assert origData[0].shape==self.full_img_size, \
            "Shape of data is wrong "+str(origData[0].shape) +" "+str(self.full_img_size)
        #get labels : multi-label: per image: a value for each word in dictionary
        labels_mid = self.get_labels_from_file(batch)
        assert len(labels_mid) == len(origData), \
            "Data and Labels do not match in length "+str(len(labels_mid))+','+str(len(origData))
        #now: grab patches/resize to the self.inner_size and keep labels associated correctly
        #find area of interest Stored in dictionary with data, labels, dictionary
        fixCrop = self.get_crop_from_file(batch)
        #get patches of image, correctly labeled
        data, labels, locs = self.get_patched_data_from_file(origData, labels_mid, fixCrop)
        #shuffle the above to mix them up a little more?
        ################TODO : if we are plotting we don't want this...
        print "Returning Batch"
        #self.debug_print(data, labels)
        return (data, labels, locs), origData#[:,:,:,::-1]

    def pt_in_box(self, location, box):
        return (location[0]<box[2] and location[0]>box[0] and location[1]>box[1] and location[1]<box[3]) or \
            (location[0]+location[2]<box[2] and location[0]+location[2]>box[0] and location[1]+location[2]>box[1] and location[1]+location[2]<box[3]) or \
            (location[0]<box[2] and location[0]>box[0] and location[1]+location[2]>box[1] and location[1]+location[2]<box[3]) or \
            (location[0]+location[2]<box[2] and location[0]+location[2]>box[0] and location[1]>box[1] and location[1]<box[3])

    def get_batch_from_file_random(self):
        #use self.word to select labels
        #others loaded all at start, here we load a batch from file at a time...
        batch = unpickle(self.get_data_file_name(self.curr_batchnum - self.batch_range[0]))
        origData = batch['data']
        #need to parse thisBatch into proper format: pull out images and stack for training;
        #scale depth down to similar range as rest of data...
        #so: # no done before mean subtraction, scale depth portion of mean too
        if self.scale_depth:
            #scale the last dimension of every sample to be 0-255
            minVal = 0.0
            maxVal = origData[:,:,:,3].max() #max across all depth
            #assert maxVal==9870, "Max changed "+str(maxVal)+','+str(9870)
            if maxVal!=9870:
                print "Max value varies ",maxVal,',',9870
            #scaled_depth = np.round(255.0 * (depth - minVal) / (maxVal - minVal - 1.0))
            #origData[:,:,:,3] = n.round(255.0 * (origData[:,:,:,3] - minVal) / (maxVal - minVal - 1.0))
            origData[:,:,:,3] = n.round(255.0 * (origData[:,:,:,3] - minVal) / (9870.0 - minVal - 1.0))
        #pull out labels for self.word and stack for label vector 
        if not self.subtract_mean_patch:
            origData = origData-self.data_mean
        #resize if needed now
        if not self.full_resolution:
            origData = [misc.imresize(origData[i][0],(self.full_img_size[0],self.full_img_size[1]), interp='bilinear', mode=None) for i in range(0,len(origData))]
        #remove the depth if needed: after mean removed
        if not self.with_depth:
            #cut last channel of each item
            #fullData = [item[:,:,:3] for item in fullData]
            origData = origData[:,:,:,:3]
        assert origData[0].shape==self.full_img_size, \
            "Shape of data is wrong "+str(origData[0].shape) +" "+str(self.full_img_size)
        #get labels : multi-label: per image: a value for each word in dictionary
        labels_mid = self.get_labels_from_file(batch)
        assert len(labels_mid) == len(origData), \
            "Data and Labels do not match in length "+str(len(labels_mid))+','+str(len(origData))
        #now: grab patches/resize to the self.inner_size and keep labels associated correctly
        #find area of interest Stored in dictionary with data, labels, dictionary
        fixCrop = self.get_crop_from_file(batch)
        data = []
        locs = []
        labels = []
        #get patches of image, correctly labeled
        for i,image in enumerate(origData):
            datat, locst = self.multi_res_patches(image)
            locst = [item+(i,) for item in locst]
            data = data + datat
            locs = locs + locst
            labelst = [labels_mid[i] if self.pt_in_box( location, fixCrop[i] ) else len(self.batch_meta['label_names'])-1 for location in locst]
            labels = labels + labelst
        #subtract mean patch
        if self.subtract_mean_patch:
            assert data[0].shape == self.mean_patch.shape, \
                "Mean patch is different shape than the image patches "+str(self.mean_patch.shape)+","+str(finaldata[0].shape)
            data = [item-self.mean_patch for item in data]
        #shuffle the above to mix them up a little more?
        ################TODO : if we are plotting we don't want this...
        #print len(data),',',len(labels)
        #self.debug_print(data,labels)
        print "Returning Batch"
        return (data, labels, locs), origData#[:,:,:,::-1]

    def debug_print(self, data, labels):
        for i,item in enumerate(data):
            if type(labels[0])=='dict':#not self.flatlabels:
                #show label
                if labels[i]==1:
                    print "Image is ", self.word
                else:
                    print "Image is NOT ",self.word
            else:
                print "Image is ",self.batch_meta['label_names'][labels[i]]
            #show image
            #cv2.imshow('Color',item.astype(n.uint8))
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            pl.imshow(item[:,:,:3].astype(n.uint8))
            pl.show()
            pl.clf()
        emergency_quit()

    def __del__(self):
        if hasattr(self, 'streamer'):
            self.end_testing()

    def end_testing(self):
        if self.streamer.streaming:
            self.streamer.end_streaming()
        self.streamer.close_camera()

    def get_plottable_data(self, data):
        #this reshape may be wrong: check data order especially with depth info included TODO: alter for depth
        #print "Plottable ",type(data), data.shape
        data.astype(n.single)
        #problem adding mean back in...these training images are patches...
        if not self.subtract_mean_patch:
            #mean shape still an issue here, really what we need is to keep a clean copy of the data rather then trying to rebuild it.
            #above would also solve our noise problem: when we remove 20%, how do we get it back?
            assert type(data[:][0]) == type(self.data_mean), "Mean data and image data have different types "+str(type(data[:][0]))+","+str(type(self.data_mean))
            data = data.T.reshape(data.shape[1], self.num_colors, self.final_size, self.final_size).swapaxes(1,3).swapaxes(1,2)
            data = data[:,:,:,::-1]
            assert data[0].shape == self.data_mean.shape, "Mean data is different shape than the image data "+str(self.data_mean.shape)+","+str(data[0].shape)
            return n.require(data+self.data_mean[:,:,::-1], dtype=n.nuint8)[:,:,:,1:]
        else:
            assert type(data[:][0]) == type(self.mean_patch), "Mean patch and image patch have different types "+str(type(data[:][0]))+","+str(type(self.mean_patch))
            data = (data.T.reshape(data.shape[1], self.num_colors, self.final_size, self.final_size)).swapaxes(1,3).swapaxes(1,2)
            #display_color((data[0]+self.mean_patch).astype(n.uint8))
            data = data[:,:,:,::-1]
            #display_color((data[0]+self.mean_patch[:,:,::-1]).astype(n.uint8))
            assert data[0].shape == self.mean_patch.shape, "Mean patch is different shape than the image patches "+str(self.mean_patch.shape)+","+str(data[0].shape)
            return n.require(data+(self.mean_patch[:,:,::-1]), dtype=n.uint8)[:,:,:,1:]
        #is there a way to remove this reshaping? just keep the original data and return that? Remove the need to find correct mean patch, and drop out issue...

    def get_plottable_data_what(self, data, image, locs):
        newForm = []
        for i in range(0, len(locs)): #zip(data, locs):
            # grab patch from image, overwrite data[i]
            newForm.append(cv2.resize(n.require(image[locs[i][0]:locs[i][0]+locs[i][2],locs[i][1]:locs[i][1]+locs[i][2],:3], dtype=n.uint8), (self.final_size, self.final_size)))
        return n.asarray(newForm)


    '''
    Really I should stop writing this way: tests belong elsewhere, but I am not sure how to 
    restructure this entire class so that i can check input format without adding these...in 
    a unit test class, define an inheritor of this class that prints everything thta I can't 
    check with assertions? (images)
    Prints images, and cropped images to verify that format is correct across data set. 
    '''
    def check_image_crop(self, batch):
        images = batch['data']
        labels = batch['labels']
        crop_boxes = batch['crop_boxes']
        for image, label, crop_box in zip(images,labels,crop_boxes):
            print label,',',self.batch_meta['label_names'][label]
            print "Crop Box: ",crop_box
            nooo = (image.astype(n.uint8)[:,:,:3])[:,:,::-1]
            print nooo.shape
            display_color(nooo)
            print nooo[crop_box[0]:crop_box[2],crop_box[1]:crop_box[3],:].shape
            #emergency_quit()
            display_color(nooo[crop_box[0]:crop_box[2],crop_box[1]:crop_box[3],:])
            #display_color(nooo[crop_box[2]:crop_box[0],crop_box[3]:crop_box[1],:])
            emergency_quit()

    #the rest are optional (unless we decide we need them)
    #def __add_subbatch(self, batch_num, sub_batchnum, batch_dic):
    #def _join_batches(self, main_batch, sub_batch):
    #def get_batch(self, batch_num):
    #def advance_batch(self):
    #def get_next_batch_idx(self):
    #def get_next_batch_num(self):
    #def get_data_file_name(self, batchnum=None):
    #@classmethod
    #def get_instance(cls, data_dir, batch_range=None, init_epoch=1, init_batchnum=None, type="default", dp_params={}, test=False):
    #@classmethod
    #def register_data_provider(cls, name, desc, _class):
    #@staticmethod
    #def get_batch_meta(data_dir):
    #@staticmethod
    #def get_batch_filenames(srcdir):
    #@staticmethod
    #def get_batch_nums(srcdir):   
    #@staticmethod
    #def get_num_batches(srcdir):

if __name__ == "__main__":
    #nr.seed(5)
    #print "Size of a cuint8 ctype: ",ctypes.sizeof(ctypes.c_uint8)
    #print "Size of a cuint8 triplet ctype: ",ctypes.sizeof(ctypes.c_uint8 * 3)
    #print "Size of a cuint16 ctype: ",ctypes.sizeof(ctypes.c_uint16)
    ok = KinectDataProvider("data/shapetest/real/depth/", (1,4), 1, None, {}, True)
    ok.get_next_batch()
    ok.end_testing()

    '''
    #oniFrame: what we get for one frame
        dataSize = 'ctypes.c_int' //size in terms of the bytes
        data = 'ctypes.c_void_p' //data of whatever type -> array of blah
        sensorType = 'OniSensorType' //
        timestamp = 'ctypes.c_ulonglong' //
        frameIndex = 'ctypes.c_int' //frame since first streamed
        width = 'ctypes.c_int' //duh, check 480x640
        height = 'ctypes.c_int' //
        videoMode = 'OniVideoMode' //depth vs color?
        croppingEnabled = 'OniBool' // where so we set this...
        cropOriginX = 'ctypes.c_int' //
        cropOriginY = 'ctypes.c_int' //
        stride = 'ctypes.c_int' //
    '''