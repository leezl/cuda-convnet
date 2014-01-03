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

import numpy as n
import os
from time import time, asctime, localtime, strftime
from collections import deque
from numpy.random import randn, rand
from numpy import s_, dot, tile, zeros, ones, zeros_like, array, ones_like
from util import *
from data import *
from options import *
from math import ceil, floor, sqrt
from data import DataProvider, dp_types
import sys
import shutil
import platform
from os import linesep as NL

class ModelStateException(Exception):
    pass

# GPU Model interface
class IGPUModel:
    def __init__(self, model_name, op, load_dic, filename_options=None, dp_params={}):
        # these are input parameters
        self.model_name = model_name
        self.op = op
        self.options = op.options
        self.load_dic = load_dic
        self.filename_options = filename_options
        self.dp_params = dp_params
        self.get_gpus()
        self.fill_excused_options()
        #assert self.op.all_values_given()
        
        for o in op.get_options_list():
            setattr(self, o.name, o.value)

        # early stop setup
        if hasattr(self, 'early_stop_class'):
            #initial error or other setup per class
            self.min_test_error = 1.0 #initially, worst error possible: 100% wrong...
            self.alpha = 3
            if self.early_stop_class=="pq":
                #need to store k epochs of training error...
                self.k = 5 #because it was 5 in the paper?
                self.alpha = 3
                #are we counting an epoch as...each time we test == 1 epoch, 
                #even though we can set it to train on less than all info each epoch...
                #assume: that we set it correctly?
                # Assume the error before we've run anything is worst possible.
                self.epoch_average_train_errors = deque([1.0 for i in range(0, self.k)])

        # these are things that the model must remember but they're not input parameters
        if load_dic:
            self.model_state = load_dic["model_state"]
            self.save_file = self.options["load_file"].value
            if not os.path.isdir(self.save_file):
                self.save_file = os.path.dirname(self.save_file)
        else:
            self.model_state = {}
            if filename_options is not None:
                self.save_file = model_name + "_" + '_'.join(['%s_%s' % (char, self.options[opt].get_str_value()) for opt, char in filename_options]) + '_' + strftime('%Y-%m-%d_%H.%M.%S')
            self.model_state["train_outputs"] = []
            self.model_state["test_outputs"] = []
            self.model_state["epoch"] = 1
            self.model_state["batchnum"] = self.train_batch_range[0]
        self.init_data_providers()
        if load_dic: 
            self.train_data_provider.advance_batch()
            
        # model state often requries knowledge of data provider, so it's initialized after
        try:
            self.init_model_state()
        except ModelStateException, e:
            print e
            sys.exit(1)
        for var, val in self.model_state.iteritems():
            setattr(self, var, val)
            
        self.import_model()
        self.init_model_lib()
        
    def import_model(self):
        print "========================="
        print "Importing %s C++ module" % ('_' + self.model_name)
        self.libmodel = __import__('_' + self.model_name) 
                   
    def fill_excused_options(self):
        pass
    
    def init_data_providers(self):
        self.dp_params['convnet'] = self
        try:
            self.test_data_provider = DataProvider.get_instance(self.data_path, self.test_batch_range,
                                                                type=self.dp_type, dp_params=self.dp_params, test=True)
            self.train_data_provider = DataProvider.get_instance(self.data_path, self.train_batch_range,
                                                                     self.model_state["epoch"], self.model_state["batchnum"],
                                                                     type=self.dp_type, dp_params=self.dp_params, test=False)
        except DataProviderException, e:
            print "Unable to create data provider: %s" % e
            self.print_data_providers()
            sys.exit()
        
    def init_model_state(self):
        pass
       
    def init_model_lib(self):
        pass
    
    def start(self):
        if self.test_only:
            self.test_outputs += [self.get_test_error()]
            self.print_test_results()
            sys.exit(0)
        self.train()
    
    def train(self):
        print "========================="
        print "Training %s" % self.model_name
        self.op.print_values()
        print "========================="
        self.print_model_state()
        print "Running on CUDA device(s) %s" % ", ".join("%d" % d for d in self.device_ids)
        print "Current time: %s" % asctime(localtime())
        print "Saving checkpoints to %s" % os.path.join(self.save_path, self.save_file)
        print "========================="
        #gets locs from kinect data provider here..
        next_data = self.get_next_batch()
        #TODO add early stopping condition here: or blah
        while self.epoch <= self.num_epochs:
            data = next_data
            self.epoch, self.batchnum = data[0], data[1]
            self.print_iteration()
            sys.stdout.flush()
            
            compute_time_py = time()
            self.start_batch(data)
            
            # load the next batch while the current one is computing #gets locs from kinect data provider here..
            next_data = self.get_next_batch()
            
            batch_output = self.finish_batch()
            self.train_outputs += [batch_output]
            self.print_train_results()

            if self.get_num_batches_done() % self.testing_freq == 0:
                self.sync_with_host()
                self.test_outputs += [self.get_test_error()]
                self.print_test_results()
                self.print_test_status()
                #add condition here for not Saving if early stopping criteria met
                done = self.early_stop()
                if not done:
                    self.conditional_save()
                else:
                    print "NOTICE: Early Stopping Condition Met ",self.min_test_error
                    #rollback to minimum error epoch values: ?
                    #For now: only allow it to save if it has improved this epoch?
                    sys.exit()
            self.print_train_time(time() - compute_time_py)
        self.cleanup()

    def early_stop(self):
        #DEBUG
        print "Test outputs at -1: ",self.test_outputs[-1]
        #first value: sum negative log prob, second: sum error, third: numsamples this batch
        #self.test_outputs = [(logprob, error), ...]#[({'logprob': [1004.6340484619141, 278.0]}, 325), ...]
        #check what the chosen early_stopping condition is
        #if GL (generalization loss)
        if self.early_stop_class == "gl":
            #needs alpha (threshold)
            #need to store minimum test error ever... self.min_test_error
            #update min if needed:
            if self.min_test_error > self.test_outputs[-1][0]['logprob'][1]:
                self.min_test_error = self.test_outputs[-1][0]['logprob'][1]
                #save this state
                self.save_opt_state() #Current epoch settings saved
            # find gl: (((average) validation/test error this epoch / minimum test error (ever)) -1.0 ) * 100.0
            print "Min Error ", self.min_test_error
            print "Test Outputs ", self.test_outputs[-1][0]['logprob'][1]
            gl = 100 * ((self.test_outputs[-1][0]['logprob'][1] / self.min_test_error) - 1.0)
            print "gl ",gl,"   Alpha ",self.alpha
            if gl > self.alpha: #this alpha should probably be a parameter
                return True #time to stop
        elif self.early_stop_class == "pq":
            #requires and alpha and a k (needs gl and pk)
            #needs to keep track of k epochs...
            #update min if needed:
            if self.min_test_error > self.test_outputs[-1][0]['logprob'][1]:
                self.min_test_error = self.test_outputs[-1][0]['logprob'][1]
                #save this state
                self.save_opt_state() #Current epoch settings saved
            # find gl: (((average) validation/test error this epoch / minimum test error (ever)) -1.0 ) * 100.0
            gl = 100 * ((self.test_outputs[-1][0]['logprob'][1] / self.min_test_error) - 1.0)
            #add new average test error to teh self.epoch_average_train_errors
            #sum and divide the errors for this epoch #off by one?
            print type(self.train_outputs),',',self.train_outputs[0]
            tr_err = [item[0]['logprob'][1] for item in self.train_outputs[len(self.train_outputs)-self.testing_freq:len(self.train_outputs)]]
            print type(tr_err),',',type(tr_err[0]),',',tr_err[0]
            newAvg = sum(tr_err) / self.testing_freq
            print "New Train average for this epoch... ", newAvg
            self.epoch_average_train_errors.append( newAvg )
            #remove the oldest value in the list
            self.epoch_average_train_errors.popleft()
            #pk: 1000 * ((sum last k epochs training errors / (k * min training error for last k epochs)) -1)
            pk = 1000 * ( ((sum(self.epoch_average_train_errors)) / (self.k * min(self.epoch_average_train_errors))) - 1.0 )
            print "gl: ",gl," pk: ",pk,',',gl/pk
            #calculate the quotient
            if self.alpha < (gl / pk):
                return True
        #thrid class UP should be ok: stop after error increases in s consecutive strips
        elif self.early_stop_class == "up":
            print "UP early stop class not implemented, continuing."
            pass
        return False #not time to stop
    
    def cleanup(self):
        sys.exit(0)
        
    def sync_with_host(self):
        self.libmodel.syncWithHost()
            
    def print_model_state(self):
        pass
    
    def get_num_batches_done(self):
        return len(self.train_batch_range) * (self.epoch - 1) + self.batchnum - self.train_batch_range[0] + 1
    
    def get_next_batch(self, train=True):
        dp = self.train_data_provider
        if not train:
            dp = self.test_data_provider
        return self.parse_batch_data(dp.get_next_batch(), train=train)
    
    def parse_batch_data(self, batch_data, train=True):
        return batch_data[0], batch_data[1], batch_data[2]['data']
    
    def start_batch(self, batch_data, train=True):
        self.libmodel.startBatch(batch_data[2], not train)
    
    def finish_batch(self):
        return self.libmodel.finishBatch()
    
    def print_iteration(self):
        print "\t%d.%d..." % (self.epoch, self.batchnum),
    
    def print_train_time(self, compute_time_py):
        print "(%.3f sec)" % (compute_time_py)
    
    def print_train_results(self):
        batch_error = self.train_outputs[-1][0]
        if not (batch_error > 0 and batch_error < 2e20):
            print "Crazy train error: %.6f" % batch_error
            self.cleanup()

        print "Train error: %.6f " % (batch_error),

    def print_test_results(self):
        batch_error = self.test_outputs[-1][0]
        print "%s\t\tTest error: %.6f" % (NL, batch_error),

    def print_test_status(self):
        status = (len(self.test_outputs) == 1 or self.test_outputs[-1][0] < self.test_outputs[-2][0]) and "ok" or "WORSE"
        print status,
        
    def conditional_save(self):
        batch_error = self.test_outputs[-1][0]
        if batch_error > 0 and batch_error < self.max_test_err:
            self.save_state()
        else:
            print "\tTest error > %g, not saving." % self.max_test_err,
    
    def aggregate_test_outputs(self, test_outputs):
        #cannot divide by test range if streaming constantly...
        print "DEBUG aggregate ",self.test_one,',',self.test_from_camera
        test_error = tuple([sum(t[r] for t in test_outputs) / (1 if (self.test_one or self.test_from_camera) else len(self.test_batch_range)) for r in range(len(test_outputs[-1]))])
        return test_error
    
    def get_test_error(self):
        #gets locs from kinect data provider here..
        next_data = self.get_next_batch(train=False)
        test_outputs = []
        #this will go forever if the option:'test_from_camera' is active
        while True:
            data = next_data
            self.start_batch(data, train=False)
            #TODO: if streaming tests, change this check so it keeps testing until ctrl+C?
            #or self.op.get_value('stream_testing') or self.op.get_value('test_from_camera')
            load_next = (not self.test_one and data[1] < self.test_batch_range[-1]) or (self.op.get_value('test_from_camera') and not self.test_one)
            if load_next: # load next batch
                #gets locs from kinect data provider here..
                next_data = self.get_next_batch(train=False)
            test_outputs += [self.finish_batch()]
            #TODO: if we are streaming from camera, we do not want to keep test results forever. : drop this after some amount...
            if self.op.get_value('test_from_camera'):
                #drop first batch in list
                assert len(test_outputs)>1, "tets_outputs too short to drop a batch "+str(len(test_outputs))
                test_outputs.pop(0)#first one inefficient
            #if self.camera_stream: #while len(test_outputs)>30: #or some number or drop each time if we know we're streaming
            #   test_outputs.pop(0) #pop first item out
            if self.test_only: # Print the individual batch results for safety
                print "batch %d: %s" % (data[1], str(test_outputs[-1]))
            if not load_next:
                break
            sys.stdout.flush()
            #if we want to do something with the results (display, pass to another program)
            #we should send/display here: aggregate_test_outputs here/ OR see 
            #use test_outputs:
            #assume we ran on a single frame:
            #test_outputs: results for one frame...
            #test_outputs[-1] -> single batch results
            #test_outputs[-1][x] -> a single patch/image/sample result
            #this is : neg (log prob, error...?) will depend on type of network
            #DEBUG:
            print "DEBUG Test_OutPuts ",type(test_outputs),',',test_outputs[-1]
            #Test_OutPuts  <type 'list'> , ({'logprob': [1004.6340484619141, 278.0]}, 325)#logprob for all, error for all, num samples in returned batch
            #So: take label, rebuild images, (get_plotable_data())
            #new function: iterate_plotable:
            #takes plotable data and displays all of it in sets of eight or so:
            #so we can take a single camera frame and see how it labeled each patch:
            #new function in data_provider: gives locations of each patch from image
        return self.aggregate_test_outputs(test_outputs)
    
    def set_var(self, var_name, var_val):
        setattr(self, var_name, var_val)
        self.model_state[var_name] = var_val
        return var_val
        
    def get_var(self, var_name):
        return self.model_state[var_name]
        
    def has_var(self, var_name):
        return var_name in self.model_state
        
    def save_state(self):
        for att in self.model_state:
            if hasattr(self, att):
                self.model_state[att] = getattr(self, att)
        
        dic = {"model_state": self.model_state,
               "op": self.op}
            
        checkpoint_dir = os.path.join(self.save_path, self.save_file)
        checkpoint_file = "%d.%d" % (self.epoch, self.batchnum)
        checkpoint_file_full_path = os.path.join(checkpoint_dir, checkpoint_file)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    
        pickle(checkpoint_file_full_path, dic,compress=self.zip_save)
        
        for f in sorted(os.listdir(checkpoint_dir), key=alphanum_key):
            if sum(os.path.getsize(os.path.join(checkpoint_dir, f2)) for f2 in os.listdir(checkpoint_dir)) > self.max_filesize_mb*1024*1024 and f != checkpoint_file:
                os.remove(os.path.join(checkpoint_dir, f))
            else:
                break

    def save_opt_state(self):
        for att in self.model_state:
            if hasattr(self, att):
                self.model_state[att] = getattr(self, att)
        
        dic = {"model_state": self.model_state,
               "op": self.op}
            
        checkpoint_dir = os.path.join(self.save_path, self.save_file)
        checkpoint_file = "Opt_%d.%d.%d" % (self.epoch, self.batchnum, int(ceil(self.min_test_error*100)))
        checkpoint_file_full_path = os.path.join(checkpoint_dir, checkpoint_file)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    
        pickle(checkpoint_file_full_path, dic,compress=self.zip_save)
        
        for f in sorted(os.listdir(checkpoint_dir), key=alphanum_key):
            if "Opt" in f and sum(os.path.getsize(os.path.join(checkpoint_dir, f2)) for f2 in os.listdir(checkpoint_dir)) > self.max_filesize_mb*1024*1024 and f != checkpoint_file:
                os.remove(os.path.join(checkpoint_dir, f))
            else:
                break
            
    @staticmethod
    def load_checkpoint(load_dir):
        if os.path.isdir(load_dir):
            return unpickle(os.path.join(load_dir, sorted(os.listdir(load_dir), key=alphanum_key)[-1]))
        return unpickle(load_dir)

    @staticmethod
    def get_options_parser():
        op = OptionsParser()
        op.add_option("f", "load_file", StringOptionParser, "Load file", default="", excuses=OptionsParser.EXCLUDE_ALL)
        op.add_option("train-range", "train_batch_range", RangeOptionParser, "Data batch range: training")
        op.add_option("test-range", "test_batch_range", RangeOptionParser, "Data batch range: testing")
        op.add_option("data-provider", "dp_type", StringOptionParser, "Data provider", default="default")
        op.add_option("test-freq", "testing_freq", IntegerOptionParser, "Testing frequency", default=25)
        op.add_option("epochs", "num_epochs", IntegerOptionParser, "Number of epochs", default=500)
        op.add_option("data-path", "data_path", StringOptionParser, "Data path")
        op.add_option("save-path", "save_path", StringOptionParser, "Save path")
        op.add_option("max-filesize", "max_filesize_mb", IntegerOptionParser, "Maximum save file size (MB)", default=5000)
        op.add_option("max-test-err", "max_test_err", FloatOptionParser, "Maximum test error for saving")
        op.add_option("num-gpus", "num_gpus", IntegerOptionParser, "Number of GPUs", default=1)
        op.add_option("test-only", "test_only", BooleanOptionParser, "Test and quit?", default=0)
        op.add_option("zip-save", "zip_save", BooleanOptionParser, "Compress checkpoints?", default=0)
        op.add_option("test-one", "test_one", BooleanOptionParser, "Test on one batch at a time?", default=1)
        op.add_option("gpu", "gpu", ListOptionParser(IntegerOptionParser), "GPU override", default=OptionExpression("[-1] * num_gpus"))
        #add stream testing option: keep testing and send results elsewhere
        #op.add_option("stream-testing", "stream_testing", BooleanOptionParser, "Stream testing data, not batches?", default=0)
        #add early stopping criteria check:
        op.add_option("early-stop-class", "early_stop_class", StringOptionParser, "Selection for early stopping: GL,PQ, or UP (GL only for now", default="")
        return op

    @staticmethod
    def print_data_providers():
        print "Available data providers:"
        for dp, desc in dp_types.iteritems():
            print "    %s: %s" % (dp, desc)
            
    def get_gpus(self):
        self.device_ids = [get_gpu_lock(g) for g in self.op.get_value('gpu')]
        if GPU_LOCK_NO_LOCK in self.device_ids:
            print "Not enough free GPUs!"
            sys.exit()
        
    @staticmethod
    def parse_options(op):
        try:
            load_dic = None
            options = op.parse()
            if options["load_file"].value_given:
                load_dic = IGPUModel.load_checkpoint(options["load_file"].value)
                old_op = load_dic["op"]
                old_op.merge_from(op)
                op = old_op
            op.eval_expr_defaults()
            return op, load_dic
        except OptionMissingException, e:
            print e
            op.print_usage()
        except OptionException, e:
            print e
        except UnpickleError, e:
            print "Error loading checkpoint:"
            print e
        sys.exit()
        
