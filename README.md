Cuda-Convnet with Kinect
====================================

This project is based off of the one here:    
https://code.google.com/p/cuda-convnet/    

Our goal is to alter this project to grab images from a Kinect and run on them instead. Which works, but is not fast enough for me.    

The openni files cluttering this directory had to be located here since the python openni extension searches for them here, in spite of the fact that I added openni to the paths of my computer. Perhaps this is fixable, perhaps not.

Requirements:
  *primesense camera
  *OpenNI2 (only version that supports pyhton at the moment I think)
  *NiTE
  *primesense python package
  
Otherwise:
  *The all c++ code should be able to work with the kinect if it uses a pre-2.0 version.


Strange Issues:
=====================
OpenNI and Python:
    *Color Streams appear strange if they are stopped, and then restarted.   
    *Color and Depth Syncing cannot be activated   
        -> the function takes no parameters, but complains in the OpenNI code that a parameter is wrong   
    *On certain computers: Color and Depth can be read at the same time in Quarter size, (320,240, :), or if depth is (640, 480, 1), but not if color is (640, 480, 3), regardless of what depth is. Note: Color can be read at this size by itself.


NOTES
===================
* Need to update get_plottable to use correct means to fix images. Patches complicate matters.   
* Make it possible to remove depth: to check if it is resulting in better or worse behavior.   
* Find out where it is getting the labels from for plotting and make sure it finds the correct labels.   
* Create a separate meta_data for training: patches, depth, single, label...seems silly...   
* Alter data to use non-masked images: this could be contributing to test vs training problems. 
* Collect some new training data that uses the current camera to check if the difference between camera have an influence.
* Make sure the depth data is similar format...totally failed to investigate that...  
   
Maybe:   
* Create separate data set of smaller batches (assuming training and testing batches have to be similar size...)   
