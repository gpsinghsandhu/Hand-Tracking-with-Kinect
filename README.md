Hand-Tracking-with-Kinect
=========================

This is the implementation of Hand Tracking with Kinect using OpenCV. This is also my proposal to OpenCV for GSoC 2013

Summary
=======
he aim of this project is to create a framework for detecting/tracking hand in the data available from kinect (and other RGB-D sensors). The approach is inspired from the Tracking-Learning-Detecting (TLD) framework proposed by Kalal and team. The framework contains - a color+depth based tracker and an appearance based detection module. While the tracker is used to track the hand rather cheaply (in terms of computation), the detector is used to find hand in case of tracker failure due to occlusion and high speed movement. The detector module is a random-forest based classifier which is trained on the data available from the video (RGB-D) itself and does not require any proir training. Also this detector module is proposed to be trained on a separate thread. Additionally the framework also contains a module for tracking finger and palm detection for detecting finger tips (if visible) and palm center respectively. 

Here is my full propasal to OpenCV - http://www.google-melange.com/gsoc/proposal/review/google/gsoc2013/gpsinghsandhu/1
Here is the result of some preliminary implementation - https://www.youtube.com/watch?v=r3-96INmhCA
