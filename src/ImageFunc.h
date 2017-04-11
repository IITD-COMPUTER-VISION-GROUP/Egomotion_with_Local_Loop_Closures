/**
 * This file is a part of Egomotion with Local Loop Closures(ELLC) project.
 * For more information visit: http://www.cse.iitd.ac.in/~suvam/wacv2017.html
 **/

#pragma once

#ifndef __ImageFunc__
#define __ImageFunc__


#include <cstdio>
#include<ctime>
#include <cstdlib>

#include"opencv2/opencv.hpp"
#include<complex>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "DepthHypothesis.h"
#include "DepthPropagation.h"

#include <fstream>
#include "Frame.h"

using namespace std;
using namespace cv;


vector<float> GetImagePoseEstimate(frame* prev_frame, frame* current_frame, int frame_num, depthMap* currDepthMap,frame* tminus1_prev_frame,float* initial_pose_estimate, bool fromLoopClosure=false, bool homo=false);

#endif /* defined(__ImageFunc__) */
