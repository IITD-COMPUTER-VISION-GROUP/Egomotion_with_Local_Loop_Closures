/**
 * This file is a part of Egomotion with Local Loop Closures(ELLC) project.
 * For more information visit: http://www.cse.iitd.ac.in/~suvam/wacv2017.html
 **/

#ifndef __PixelWisePyramid__
#define __PixelWisePyramid__

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "Frame.h"
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <unsupported/Eigen/MatrixFunctions>
#include <cstdio>
#include<ctime>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/thread/thread.hpp>
#include <boost/date_time.hpp>
#include <pthread.h>
#include <time.h>
#include <sys/time.h>
#include <cmath>

#include "ExternVariable.h"
#include "DepthPropagation.h"
#include "DepthHypothesis.h"

using namespace cv;

/*
 The PixelWisePyramid class contains functions and variables for pose estimation using both the forward and the inverse compositional gauss newton algorithm at a particular pyramid level
 It is similar to the Pyramid class but is specifically designed to support multiple threads
 It has an option of using constant pixel weights (for which it uses inverse compositional which is faster) or variable pixel weights  (for which it uses forward compositional which is computationally more expensive)
 This implementation is as per the framework for compositional algorithm given in Section 3.1 and the inverse compositional algorithm given in Section 3.2 of 'Lucas-Kanade 20 Years On: A Unifying Framework: Part 1' available at https://www.ri.cmu.edu/pub_files/pub3/baker_simon_2002_3/baker_simon_2002_3.pdf
 */
class PixelWisePyramid{
    
public:
    
    int pyrlevel;
    int nRows,nCols;
    float* pose;
    float weightedPose;
    float prevPose[6];
    float covarianceDiagonalWts[6];
    
    Mat covarianceMatrixInv;
    Mat motionPrior;
    
    Mat hessianInv;
    Mat saveImg;
    Mat deltapose;
    Mat sd_param;
    Mat hessian;
    Mat steepestDescent; //6x(total points)
    Mat weightedSteepestDescent; //6x(total points)
    Mat savedWarpedPointsX;
    Mat savedWarpedPointsY;
    
    
    frame* prev_frame;
    frame* current_frame;
    depthMap* currentDepthMap;
    
    //for display
    Mat display_warpedimg;
    Mat display_templateimg;
    Mat display_2bewarpedimg;
    Mat display_iterationres;
    Mat display_origres;
    Mat display_weightimg;
    
    //multi-thread variables
    //if using more than 3 threads, then need to modify this !!!!
    Mat hessian_thread1;
    Mat hessian_thread2;
    Mat hessian_thread3;
    
    Mat sd_param_thread1;
    Mat sd_param_thread2;
    Mat sd_param_thread3;
    
    Mat test_img;
    
public:
    
    PixelWisePyramid(frame* prevframe,frame* currentframe,float* pose,depthMap* currDepthMap);

    void putPreviousPose(frame* tminus1_prev_frame);
    
    void updatePose();
    void saveWeights(bool useAverageWeights=false);
    
    void calculatePixelWise(int ymin,int ymax, int thread_num); //parallelized
    void calculatePixelWiseParallel();
    
    void iteratePixelWiseInvCompositional(int ymin,int ymax, int thread_num);//parallelized 
    void precomputePixelWiseInvCompositional(int ymin,int ymax, int thread_num);//parallelized
    void calculatePixelWiseParallelInvCompositional(int iter);
   
    
    ~PixelWisePyramid();
    
    
};

#endif /* defined(__PixelWisePyramid__) */
