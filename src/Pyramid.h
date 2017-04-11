/**
 * This file is a part of Egomotion with Local Loop Closures(ELLC) project.
 * For more information visit: http://www.cse.iitd.ac.in/~suvam/wacv2017.html
 **/

#ifndef __Pyramid__
#define __Pyramid__

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "Frame.h"
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <unsupported/Eigen/MatrixFunctions>

#include <cstdio>
#include <ctime>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include "ExternVariable.h"
#include <cmath>
#include "DepthPropagation.h"
#include "DepthHypothesis.h"



using namespace cv;


/*
The Pyramid class contains functions and variables for pose estimation using the forward compositional gauss newton algorithm at a particular pyramid level
 This implementation is as per the framework for compositional algorithm given in Section 3.1 of 'Lucas-Kanade 20 Years On: A Unifying Framework: Part 1' available at https://www.ri.cmu.edu/pub_files/pub3/baker_simon_2002_3/baker_simon_2002_3.pdf
*/
class Pyramid{
    
public:
    
    int level;
    float *pose;
    float lastErr;
    float error;
    float pointUsage;
    float weightedPose;
    float prevPose[6];
    float covarianceDiagonalWts[6];
    
    
    Mat steepestDescent;
    Mat hessianInv;
    Mat worldPoints;
    Mat transformedWorldPoints;
    Mat saveImg;
    Mat warpedPoints;
    Mat warpedImage;
    Mat residual;
    Mat warpedGradientx;
    Mat warpedGradienty;
    Mat weights;
    Mat covarianceMatrixInv;
    Mat motionPrior;
    Mat deltapose;

    
    frame* prev_frame;
    frame* current_frame;
    depthMap* currentDepthMap;
    
public:
    
    Pyramid(frame* prevframe,frame* currentframe,float* pose,depthMap* currDepthMap);
    
    void performPrecomputation();
    float performIterationSteps();
    void calculateSteepestDescent();
    void calculateHessianInv();
    void calCovarianceMatrixInv(float* covar_wts);
    void calculateWorldPoints();
    void calculateWarpedPoints();
    void calculateWarpedImage();
    void updatePose();
    void putPreviousPose(frame* tminus1_prev_frame);
    
    float calResidualAndWeights();
    void calMotionPrior();
    
    void calculatePixelWise();
    
    
};

#endif /* defined(__Pyramid__) */
