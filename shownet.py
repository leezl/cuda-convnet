# Copyright (c) 2011, Alex Krizhevsky (akrizhevsky@gmail.com)
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# 
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy
import sys, time
import getopt as opt
from util import *
from math import sqrt, ceil, floor
import os
from gpumodel import IGPUModel
import random as r
import numpy.random as nr
from convnet import ConvNet
from options import *

try:
    import pylab as pl
except:
    print "This script requires the matplotlib python library (Ubuntu/Fedora package name python-matplotlib). Please install it."
    sys.exit(1)

class ShowNetError(Exception):
    pass

class ShowConvNet(ConvNet):
    def __init__(self, op, load_dic):
        ConvNet.__init__(self, op, load_dic)
    
    def get_gpus(self):
        self.need_gpu = self.op.get_value('show_preds') or self.op.get_value('write_features') or self.op.get_value('write_features_stream') \
            or self.op.get_value('show_preds_patch') or self.op.get_value('show_preds_patch_total')
        if self.need_gpu:
            ConvNet.get_gpus(self)
    
    def init_data_providers(self):
        class Dummy:
            def advance_batch(self):
                pass
        if self.need_gpu:
            ConvNet.init_data_providers(self)
        else:
            self.train_data_provider = self.test_data_provider = Dummy()
    
    def import_model(self):
        if self.need_gpu:
            ConvNet.import_model(self)
            
    def init_model_state(self):
        #ConvNet.init_model_state(self)
        if self.op.get_value('show_preds'):
            self.sotmax_idx = self.get_layer_idx(self.op.get_value('show_preds'), check_type='softmax')
        if self.op.get_value('show_preds_patch'):
            self.sotmax_idx = self.get_layer_idx(self.op.get_value('show_preds_patch'), check_type='softmax')
        if self.op.get_value('show_preds_patch_total'):
            self.sotmax_idx = self.get_layer_idx(self.op.get_value('show_preds_patch_total'), check_type='softmax')
        if self.op.get_value('write_features'):
            self.ftr_layer_idx = self.get_layer_idx(self.op.get_value('write_features'))
        if self.op.get_value('write_features_stream'):
            self.ftr_layer_idx = self.get_layer_idx(self.op.get_value('write_features_stream'))
            
    def init_model_lib(self):
        if self.need_gpu:
            ConvNet.init_model_lib(self)

    def plot_cost(self):
        if self.show_cost not in self.train_outputs[0][0]:
            raise ShowNetError("Cost function with name '%s' not defined by given convnet." % self.show_cost)
        train_errors = [o[0][self.show_cost][self.cost_idx] for o in self.train_outputs]
        test_errors = [o[0][self.show_cost][self.cost_idx] for o in self.test_outputs]

        numbatches = len(self.train_batch_range)
        test_errors = numpy.row_stack(test_errors)
        test_errors = numpy.tile(test_errors, (1, self.testing_freq))
        test_errors = list(test_errors.flatten())
        test_errors += [test_errors[-1]] * max(0,len(train_errors) - len(test_errors))
        test_errors = test_errors[:len(train_errors)]

        numepochs = len(train_errors) / float(numbatches)
        pl.figure(1)
        x = range(0, len(train_errors))
        pl.plot(x, train_errors, 'k-', label='Training set')
        pl.plot(x, test_errors, 'r-', label='Test set')
        pl.legend()
        ticklocs = range(numbatches, len(train_errors) - len(train_errors) % numbatches + 1, numbatches)
        epoch_label_gran = int(ceil(numepochs / 20.)) # aim for about 20 labels
        epoch_label_gran = int(ceil(float(epoch_label_gran) / 10) * 10) # but round to nearest 10
        ticklabels = map(lambda x: str((x[1] / numbatches)) if x[0] % epoch_label_gran == epoch_label_gran-1 else '', enumerate(ticklocs))

        pl.xticks(ticklocs, ticklabels)
        pl.xlabel('Epoch')
        #pl.ylabel(self.show_cost)
        pl.title(self.show_cost)
        
    def make_filter_fig(self, filters, filter_start, fignum, _title, num_filters, combine_chans):
        FILTERS_PER_ROW = 16
        MAX_ROWS = 16
        MAX_FILTERS = FILTERS_PER_ROW * MAX_ROWS
        num_colors = filters.shape[0]
        f_per_row = int(ceil(FILTERS_PER_ROW / float(1 if combine_chans else num_colors)))
        filter_end = min(filter_start+MAX_FILTERS, num_filters)
        filter_rows = int(ceil(float(filter_end - filter_start) / f_per_row))
    
        filter_size = int(sqrt(filters.shape[1]))
        fig = pl.figure(fignum)
        fig.text(.5, .95, '%s %dx%d filters %d-%d' % (_title, filter_size, filter_size, filter_start, filter_end-1), horizontalalignment='center') 
        num_filters = filter_end - filter_start
        if not combine_chans:
            bigpic = n.zeros((filter_size * filter_rows + filter_rows + 1, filter_size*num_colors * f_per_row + f_per_row + 1), dtype=n.single)
        else:
            bigpic = n.zeros((3, filter_size * filter_rows + filter_rows + 1, filter_size * f_per_row + f_per_row + 1), dtype=n.single)
    
        for m in xrange(filter_start,filter_end ):
            filter = filters[:,:,m]
            y, x = (m - filter_start) / f_per_row, (m - filter_start) % f_per_row
            if not combine_chans:
                for c in xrange(num_colors):
                    filter_pic = filter[c,:].reshape((filter_size,filter_size))
                    bigpic[1 + (1 + filter_size) * y:1 + (1 + filter_size) * y + filter_size,
                           1 + (1 + filter_size*num_colors) * x + filter_size*c:1 + (1 + filter_size*num_colors) * x + filter_size*(c+1)] = filter_pic
            else:
                filter_pic = filter.reshape((3, filter_size,filter_size))
                bigpic[:,
                       1 + (1 + filter_size) * y:1 + (1 + filter_size) * y + filter_size,
                       1 + (1 + filter_size) * x:1 + (1 + filter_size) * x + filter_size] = filter_pic
                
        pl.xticks([])
        pl.yticks([])
        if not combine_chans:
            pl.imshow(bigpic, cmap=pl.cm.gray, interpolation='nearest')
        else:
            bigpic = bigpic.swapaxes(0,2).swapaxes(0,1)
            pl.imshow(bigpic, interpolation='nearest')        
        
    def plot_filters(self):
        filter_start = 0 # First filter to show
        layer_names = [l['name'] for l in self.layers]
        if self.show_filters not in layer_names:
            raise ShowNetError("Layer with name '%s' not defined by given convnet." % self.show_filters)
        layer = self.layers[layer_names.index(self.show_filters)]
        filters = layer['weights'][self.input_idx]
        if layer['type'] == 'fc': # Fully-connected layer
            num_filters = layer['outputs']
            channels = self.channels
        elif layer['type'] in ('conv', 'local'): # Conv layer
            num_filters = layer['filters']
            channels = layer['filterChannels'][self.input_idx]
            if layer['type'] == 'local':
                filters = filters.reshape((layer['modules'], layer['filterPixels'][self.input_idx] * channels, num_filters))
                filter_start = r.randint(0, layer['modules']-1)*num_filters # pick out some random modules
                filters = filters.swapaxes(0,1).reshape(channels * layer['filterPixels'][self.input_idx], num_filters * layer['modules'])
                num_filters *= layer['modules']

        filters = filters.reshape(channels, filters.shape[0]/channels, filters.shape[1])
        # Convert YUV filters to RGB
        if self.yuv_to_rgb and channels == 3:
            R = filters[0,:,:] + 1.28033 * filters[2,:,:]
            G = filters[0,:,:] + -0.21482 * filters[1,:,:] + -0.38059 * filters[2,:,:]
            B = filters[0,:,:] + 2.12798 * filters[1,:,:]
            filters[0,:,:], filters[1,:,:], filters[2,:,:] = R, G, B
        combine_chans = not self.no_rgb and channels == 3
        
        # Make sure you don't modify the backing array itself here -- so no -= or /=
        filters = filters - filters.min()
        filters = filters / filters.max()

        self.make_filter_fig(filters, filter_start, 2, 'Layer %s' % self.show_filters, num_filters, combine_chans)
    
    def plot_predictions(self):
        data = self.get_next_batch(train=False)[2] # get a test batch
        num_classes = self.test_data_provider.get_num_classes()
        NUM_ROWS = 2
        NUM_COLS = 4
        NUM_IMGS = NUM_ROWS * NUM_COLS
        NUM_TOP_CLASSES = min(num_classes, 4) # show this many top labels
        
        label_names = self.test_data_provider.batch_meta['label_names']
        if self.only_errors:
            preds = n.zeros((data[0].shape[1], num_classes), dtype=n.single)
        else:
            preds = n.zeros((NUM_IMGS, num_classes), dtype=n.single)
            rand_idx = nr.randint(0, data[0].shape[1], NUM_IMGS)
            data[0] = n.require(data[0][:,rand_idx], requirements='C')
            data[1] = n.require(data[1][:,rand_idx], requirements='C')
        data += [preds]

        # Run the model
        self.libmodel.startFeatureWriter(data, self.sotmax_idx)
        self.finish_batch()
        
        fig = pl.figure(3)
        fig.text(.4, .95, '%s test case predictions' % ('Mistaken' if self.only_errors else 'Random'))
        if self.only_errors:
            err_idx = nr.permutation(n.where(preds.argmax(axis=1) != data[1][0,:])[0])[:NUM_IMGS] # what the net got wrong
            data[0], data[1], preds = data[0][:,err_idx], data[1][:,err_idx], preds[err_idx,:]
            
        data[0] = self.test_data_provider.get_plottable_data(data[0])
        for r in xrange(NUM_ROWS):
            for c in xrange(NUM_COLS):
                img_idx = r * NUM_COLS + c
                if data[0].shape[0] <= img_idx:
                    break
                pl.subplot(NUM_ROWS*2, NUM_COLS, r * 2 * NUM_COLS + c + 1)
                pl.xticks([])
                pl.yticks([])
                img = data[0][img_idx,:,:,:]
                pl.imshow(img, interpolation='nearest')
                true_label = int(data[1][0,img_idx])

                img_labels = sorted(zip(preds[img_idx,:], label_names), key=lambda x: x[0])[-NUM_TOP_CLASSES:]
                pl.subplot(NUM_ROWS*2, NUM_COLS, (r * 2 + 1) * NUM_COLS + c + 1, aspect='equal')

                ylocs = n.array(range(NUM_TOP_CLASSES)) + 0.5
                height = 0.5
                width = max(ylocs)
                pl.barh(ylocs, [l[0]*width for l in img_labels], height=height, \
                        color=['r' if l[1] == label_names[true_label] else 'b' for l in img_labels])
                pl.title(label_names[true_label])
                pl.yticks(ylocs + height/2, [l[1] for l in img_labels])
                pl.xticks([width/2.0, width], ['50%', ''])
                pl.ylim(0, ylocs[-1] + height*2)

    #create new function similar to above: instead of plotting each patch and it's predictions: plot whole image, and several patches?
    '''
    Plot whole image followed by some random patches from that image? Try to trace the winning class in the image?
    '''
    '''
    Function Plan:
    Input: Data, Labels, Locations, full images in this batch
    We want: Collect the patches from same image: sort by imgidx
    Display full image, Display patches and classifications
        *To limit the display: ignore None for now or select random 
    * to limit computation time, select a random image? Just classify patches from that image
    *or: iterate: display eahc image and patches: long time
    @@@Split into 2 functions: one does random display, one does iterative@@@
    '''
    def plot_patch_predictions(self):
        #locations: (x,y,res,imgidx), images[imgidx]
        #currently identical to above, needs altering...
        stuff = self.get_next_batch(train=False)
        data = stuff[2] # get a test batch # data, labels
        locs = n.asarray(stuff[3]) # (upperleftx, upperlefty, size, imgidx)
        image_list = stuff[4]
        #stuff: (epoch, batchnum, [data, labels], locs, images)
        num_classes = self.test_data_provider.get_num_classes()
        #following defines how many sample predictions to display
        NUM_ROWS = 2
        NUM_COLS = 4
        NUM_IMGS = NUM_ROWS * NUM_COLS
        #above reset for numPatches
        NUM_TOP_CLASSES = min(num_classes, 4) # show this many top labels
        #assign colors to the classes: use to trace the object in the image(s)

        #random image selection
        chosen_image_idx = r.randint(0,len(image_list)-1)

        #print chosen_image_idx
        label_names = self.test_data_provider.batch_meta['label_names']
        for item in label_names:
            print "Label ",item

        if self.only_errors:
            preds = n.zeros((data[0].shape[1], num_classes), dtype=n.single)
        else:
            #instead grab all patches for single random image:
            #collect all patches where loc[3] == chosen_image_idx
            select_idx = [i for i,location in enumerate(locs) if location[3]==chosen_image_idx]
            if len(select_idx) > 12:
                rand_idx = nr.randint(0,len(select_idx),12)
                select_idx = n.asarray(select_idx)[rand_idx]
            #print "How many to display? ", len(select_idx)
            #set rows and cols to divide images evenly...:
            NUM_COLS = 4
            NUM_ROWS = (len(select_idx)/NUM_COLS) #+ 3 #for the overall image
            #print "NUM_ROWS: ",NUM_ROWS

            for i in range(0, (len(select_idx)%NUM_COLS)):
                select_idx.pop()
            data[0] = n.require(data[0][:,select_idx], requirements='C')
            data[1] = n.require(data[1][:,select_idx], requirements='C')
            locs = locs[select_idx] #keep correct locs too
            NUM_IMGS = len(select_idx)
            preds = n.zeros((NUM_IMGS, num_classes), dtype=n.single)
            #keep all patches in image(s)?
        data += [preds]
        #data = [data, labels, preds]

        # Run the model
        startRun = time.time()
        self.libmodel.startFeatureWriter(data, self.sotmax_idx)
        self.finish_batch()
        print "Run time: ",time.time()-startRun
        
        fig = pl.figure(3)
        fig.text(.4, .95, '%s test case predictions' % ('Mistaken' if self.only_errors else 'Random'))
        if self.only_errors:
            err_idx = nr.permutation(n.where(preds.argmax(axis=1) != data[1][0,:])[0])[:NUM_IMGS] # what the net got wrong
            data[0], data[1], preds = data[0][:,err_idx], data[1][:,err_idx], preds[err_idx,:]
            locs = locs[err_idx]

        #can we replace this? We have the full images, just grab the correct patches, no mean, no scaling, no drop out...
        #data[0] = self.test_data_provider.get_plottable_data(data[0])
        data[0] = self.test_data_provider.get_plottable_data_what(data[0], image_list[chosen_image_idx], locs)

        #grab image that matches following patches
        # grab image, change type, cut depth, reverse bgr to rgb
        img = ((image_list[chosen_image_idx].astype(n.uint8))[:,:,:3])[:,:,::-1]
        #show image (invisible otherwise?)
        pl.imshow(img, interpolation='nearest')
        #not showing labels for total image, so only NUM_ROWS, not *2
        pl.show()
        pl.clf()

        print "DEBUG ",NUM_ROWS,',',NUM_COLS

        for row in xrange(NUM_ROWS):#3 for full size image
            for c in xrange(NUM_COLS):
                img_idx = row * NUM_COLS + c
                if data[0].shape[0] <= img_idx:
                    break
                pl.subplot(NUM_ROWS*2, NUM_COLS, row * 2 * NUM_COLS + c + 1)
                pl.xticks([])
                pl.yticks([])
                img = data[0][img_idx,:,:,:][:,:,::-1]
                pl.imshow(img, interpolation='nearest')
                true_label = int(data[1][0,img_idx])

                img_labels = sorted(zip(preds[img_idx,:], label_names), key=lambda x: x[0])[-NUM_TOP_CLASSES:]
                pl.subplot(NUM_ROWS*2, NUM_COLS, (row * 2 + 1) * NUM_COLS + c + 1, aspect='equal')

                ylocs = n.array(range(NUM_TOP_CLASSES)) + 0.5
                height = 0.5
                width = max(ylocs)
                pl.barh(ylocs, [l[0]*width for l in img_labels], height=height, \
                        color=['r' if l[1] == label_names[true_label] else 'b' for l in img_labels])
                pl.title(label_names[true_label])
                pl.yticks(ylocs + height/2, [l[1] for l in img_labels])
                pl.xticks([width/2.0, width], ['50%', ''])
                pl.ylim(0, ylocs[-1] + height*2)

    '''
    Use this function to work on overall prediction: ignore low probability, combine probabilities for patches
    '''
    def plot_patch_predictions_total(self):
        #locations: (x,y,res,imgidx), images[imgidx]
        #currently identical to above, needs altering...
        stuff = self.get_next_batch(train=False)
        data = stuff[2] # get a test batch # data, labels
        locs = n.asarray(stuff[3]) # (upperleftx, upperlefty, size, imgidx)
        image_list = stuff[4]
        #stuff: (epoch, batchnum, [data, labels], locs, images)
        num_classes = self.test_data_provider.get_num_classes()

        #random image selection (to be removed later...display all, or predict all)
        chosen_image_idx = r.randint(0,len(image_list)-1)

        #grab possible labels
        label_names = self.test_data_provider.batch_meta['label_names']

        #grab all patches for single random image:
        #collect all patches where loc[3] == chosen_image_idx
        select_idx = [i for i,location in enumerate(locs) if location[3]==chosen_image_idx]
        #grab images, labels, and locs
        data[0] = n.require(data[0][:,select_idx], requirements='C')
        data[1] = n.require(data[1][:,select_idx], requirements='C')
        locs = locs[select_idx] #keep correct locs too
        # grab enough space to store the results into
        preds = n.zeros((len(select_idx), num_classes), dtype=n.single)
        # add space for results, this is how we get them from FeatureWriter
        data += [preds]
        #data = [data, labels, preds]

        # Run the model
        self.libmodel.startFeatureWriter(data, self.sotmax_idx)
        self.finish_batch()
        #preds should now contain array patches*classes of probabilities
        #get original image data back
        data[0] = self.test_data_provider.get_plottable_data_what(data[0], image_list[chosen_image_idx], locs)

        # find the "best" overall prediction for each pixel
        image_vote_mask = self.create_label_mask(image_list[chosen_image_idx].shape, data, locs, num_classes)
        #display each label region in the image : this will be messy
        self.display_each_label_region(image_list[chosen_image_idx], image_vote_mask.reshape(image_vote_mask.shape[0],image_vote_mask.shape[1],1), label_names)

    def display_image(self, image, label):
        #use matplotlib to display given image
        #following may not be necessary
        print "Displaying one of the test labels "
        fig = pl.figure(3)
        fig.text(.4, .95, '%s test case predictions' % (label))

        #show image (invisible otherwise?)
        pl.imshow(image[:,:,:3].astype(n.uint8)[:,:,::-1], interpolation='nearest')
        #not showing labels for total image, so only NUM_ROWS, not *2
        pl.show()
        #pl.clf()

    # function to apply the mask to the image
    def display_each_label_region(self, image, mask_image, label_names):
        def my_mask(image, mask, i):
            if mask == i:
                return image
            else:
                return 0.0
        #vectorize the above; seems this is faster than the numpy.mask stuff, which is meant for more complex stuff
        vmaskfunc = n.vectorize(my_mask)
        # create an array representing the image: x*y*numClasses
        for i in range(0, len(label_names)):
            #check if i is in our mask: display just that portion of the image/ outline...portion
            if i in mask_image:
                #mask out image where mask_imge != i
                #currentMask = (mask_image == i)
                #assert currentMask.shape == mask_image.shape, "Our mask is the wrong shape"
                result = vmaskfunc(image, mask_image, i)
                #display result labeled as correct class
                self.display_image(result, label_names[i])

    # takes image_shape, patches, locations
    # returns the "mask"
    def create_label_mask(self, image_shape, data, locs, num_classes):
        #
        # create voting array for now (x, y, num_classes)
        imageVote = n.zeros((image_shape[0],image_shape[1], num_classes), dtype=n.single)
        counter = n.ones((image_shape[0],image_shape[1], 1), dtype=n.single)
        # create voting patches
        votes = n.ones((data[0].shape[0],data[0].shape[1],data[0].shape[2],data[2].shape[1]), dtype=n.single)
        for item, prediction, loc in zip(votes, data[2], locs):
            item = item * prediction.reshape((1,1,prediction.shape[0]))#n.random.random_sample((1,1,num_classes))
            #check: votes[:,:,n] all values should be equal
            for i in range(0, num_classes): #probably a faster way to do this...
                assert n.all(item[:,:,i] == item[0,0,i]), "Probability not the same across patch for this class"
            # assign probabilities in image voter
            imageVote[loc[0]:loc[0]+loc[2],loc[1]:loc[1]+loc[2],:] = \
                imageVote[loc[0]:loc[0]+loc[2],loc[1]:loc[1]+loc[2],:] + item
            #add up how many patches we've counted at each location
            counter[loc[0]:loc[0]+loc[2],loc[1]:loc[1]+loc[2],:] = \
                counter[loc[0]:loc[0]+loc[2],loc[1]:loc[1]+loc[2],:] + 1.0
        
        # SO: we have a 
        # * counter where each location is how many patches we've used at that location
        # * image where each location is the probability sum across some number of patches...
        # First Try: average: image / counter (broadcast across num_classes dimension)
        imageAverage = n.divide(imageVote, counter)
        #assert imageAverage.shape == image_shape, "Shapes of image and average not correct: "+str(imageAverage.shape)
        # * now image is probability per class...0-1, so we take the max and assign that index as the label
        #imageMax = n.max(imageAverage, axis = 2) # gives us the max values, need indices of these...
        imageMax = n.argmax(imageAverage, axis = 2)
        #assert imageMax.shape == (imageAverage.shape[0], imageAverage.shape[1]), \
        #    "Shapes of final max and average not correct: "+str(imageMax.shape)
        print imageMax.shape
        #return the image of labels for each pixel
        return imageMax
    
    #need alternate version of this for if the send-features is enabled
    def do_write_features(self):
        if not os.path.exists(self.feature_path):
            os.makedirs(self.feature_path)
        next_data = self.get_next_batch(train=False)
        b1 = next_data[1]
        num_ftrs = self.layers[self.ftr_layer_idx]['outputs']
        while True:
            batch = next_data[1]
            data = next_data[2]
            ftrs = n.zeros((data[0].shape[1], num_ftrs), dtype=n.single)
            self.libmodel.startFeatureWriter(data + [ftrs], self.ftr_layer_idx)
            
            # load the next batch while the current one is computing
            next_data = self.get_next_batch(train=False)
            self.finish_batch()
            path_out = os.path.join(self.feature_path, 'data_batch_%d' % batch)
            pickle(path_out, {'data': ftrs, 'labels': data[1]})
            print "Wrote feature file %s" % path_out
            if next_data[1] == b1:
                break
        pickle(os.path.join(self.feature_path, 'batches.meta'), {'source_model':self.load_file,
                                                                 'num_vis':num_ftrs})

    #need alternate version of this for if the send-features is enabled
    def do_write_features_stream(self):
        if not self.test_from_camera:
            print "Error: this option requires camera streaming"
        next_data = self.get_next_batch(train=False)
        num_ftrs = self.layers[self.ftr_layer_idx]['outputs']
        while True:
            startTime = time.time()
            batch = next_data[1]
            data = next_data[2]
            ftrs = n.zeros((data[0].shape[1], num_ftrs), dtype=n.single)
            self.libmodel.startFeatureWriter(data + [ftrs], self.ftr_layer_idx)
            
            # load the next batch while the current one is computing
            next_data = self.get_next_batch(train=False)
            self.finish_batch()
            #path_out = os.path.join(self.feature_path, 'data_batch_%d' % batch)
            print "features ",len(ftrs),',',len(data[1])
            #pickle(path_out, {'data': ftrs, 'labels': data[1]})
            print "Wrote feature file"
            endTime = time.time()
            print "Time for frame: ",endTime-startTime
        #pickle(os.path.join(self.feature_path, 'batches.meta'), {'source_model':self.load_file,
        #                                                         'num_vis':num_ftrs})
           
    #need to alter following to be continuous, not one go?     
    def start(self):
        self.op.print_values()
        if self.show_cost:
            self.plot_cost()
            pl.show()
        if self.show_filters:
            self.plot_filters()
            pl.show()
        if self.show_preds:
            self.plot_predictions()
            pl.show()
        if self.show_preds_patch:
            self.plot_patch_predictions()
            pl.show()
        if self.show_preds_patch_total:
            self.plot_patch_predictions_total()
        if self.write_features:
            self.do_write_features()
            pl.show()
        if self.write_features_stream:
            self.do_write_features_stream()
            pl.show()
        sys.exit(0)
            
    @classmethod
    def get_options_parser(cls):
        op = ConvNet.get_options_parser()
        for option in list(op.options):
            if option not in ('gpu', 'load_file', 'train_batch_range', 'test_batch_range', 'cam_test', 'test_one'):
                op.delete_option(option)
        op.add_option("show-cost", "show_cost", StringOptionParser, "Show specified objective function", default="")
        op.add_option("show-filters", "show_filters", StringOptionParser, "Show learned filters in specified layer", default="")
        op.add_option("input-idx", "input_idx", IntegerOptionParser, "Input index for layer given to --show-filters", default=0)
        op.add_option("cost-idx", "cost_idx", IntegerOptionParser, "Cost function return value index for --show-cost", default=0)
        op.add_option("no-rgb", "no_rgb", BooleanOptionParser, "Don't combine filter channels into RGB in layer given to --show-filters", default=False)
        op.add_option("yuv-to-rgb", "yuv_to_rgb", BooleanOptionParser, "Convert RGB filters to YUV in layer given to --show-filters", default=False)
        op.add_option("channels", "channels", IntegerOptionParser, "Number of channels in layer given to --show-filters (fully-connected layers only)", default=0)
        op.add_option("show-preds", "show_preds", StringOptionParser, "Show predictions made by given softmax on test set", default="")
        op.add_option("only-errors", "only_errors", BooleanOptionParser, "Show only mistaken predictions (to be used with --show-preds)", default=False, requires=['show_preds'])
        op.add_option("write-features", "write_features", StringOptionParser, "Write test data features from given layer", default="", requires=['feature-path'])
        op.add_option("feature-path", "feature_path", StringOptionParser, "Write test data features to this path (to be used with --write-features)", default="")
        #add options here for streaming from the camera: send the results rather then save to file?
        op.add_option("show-preds-patch","show_preds_patch", StringOptionParser, "Show patch predictions made by given softmax on test set", default="")
        op.add_option("show-preds-patch-total","show_preds_patch_total", StringOptionParser, "Show patch predictions made by given softmax on test set in total image", default="")
        op.add_option("write-features-stream", "write_features_stream", StringOptionParser, "Stream test data features from given layer", default="", requires=['cam_test'])
        op.add_option("cam-test", "test_from_camera", BooleanOptionParser, "Get Test Batches from OpenNI Device?", default=0)#0? can i leave it False?
        #op.add_option("send-features", "send_features", StringOptionParser, "Send the test data features (probabilities) from given layer", default="", requires=['receiver-path'])
        #op.add_option("receiver-path", "receiver_path", StringOptionParser, "Send test data features to this (address?) (use with --send-features)", default="")
        #the above options should trigger an option in the net: test-one, and test-from-camera?: unlimited test batches, continuous until ctrl+c...
        
        op.options['load_file'].default = None
        return op
    
if __name__ == "__main__":
    try:
        op = ShowConvNet.get_options_parser()
        op, load_dic = IGPUModel.parse_options(op)
        model = ShowConvNet(op, load_dic)
        startTime = time.time()
        model.start()
        endTime = time.time()
    except (UnpickleError, ShowNetError, opt.GetoptError), e:
        print "----------------"
        print "Error:"
        print e 
