/**
 * This file is a part of Egomotion with Local Loop Closures(ELLC) project.
 * For more information visit: http://www.cse.iitd.ac.in/~suvam/wacv2017.html
 **/

#ifndef __LoopFrame__
#define __LoopFrame__

#include <cstdio>
#include<ctime>
#include <cstdlib>
#include"opencv2/opencv.hpp"
#include<complex>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>
#include "Frame.h"
#include "DepthPropagation.h"

using namespace cv;
using namespace std;

struct loopFrame
{
    Mat image_histogram;
    Mat image;
    int frameId=-1;
    float poseWrtWorld[6];
    float poseWrtOrigin[6];
    
    bool isValid=false;
    bool isStray=false;
    frame* this_frame;
    depthMap* this_currentDepthMap;
    
};



#endif /* defined(__LoopFrame__) */
