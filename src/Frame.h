/**
 * This file is a part of Egomotion with Local Loop Closures(ELLC) project.
 * For more information visit: http://www.cse.iitd.ac.in/~suvam/wacv2017.html
 **/

#pragma once

#ifndef __Frame__
#define __Frame__

#pragma once

#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <unsupported/Eigen/MatrixFunctions>

#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "DepthHypothesis.h"
#include "ExternVariable.h"


using namespace std;
using namespace cv;

/*
    Frame class contains input image, estimated depth map and variables associated with each frame
    formed everytime there a new incoming frame
*/

class frame
{
    
public:
    
    int frameid_new[3]; //not used
    
    
    //constructors
    frame();
    frame(VideoCapture vidcap_rgb);
    frame(VideoCapture vidcap_rgb,VideoCapture vidcap_depth); //used when depth map there is an input depth map, not sure if complete implementation still exits
    
    int no_points_substantial_grad;
    
    int frameId;

    int parentKeyframeId;
    
    bool isKeyframe; 
    
    bool deptharrfirstcall;
    
    bool deptharrpyrfirstcall;
    
    int numWeightsAdded[util::MAX_PYRAMID_LEVEL];
    
    frame* keyFrame;
    
    int width,height;
    
    int* correspondingDepthHypothesis;
    
    Mat rgb_image;
    
    Mat image; // black and white image
    
    Mat image_pyramid[util::MAX_PYRAMID_LEVEL];
    
    Mat depth;
    
    Mat weight_pyramid[util::MAX_PYRAMID_LEVEL];
    
    float* deptharr;
    
    float* deptharrpyr[util::MAX_PYRAMID_LEVEL];
    
    float* depthvararr;
    
    float* depthvararrpyr[util::MAX_PYRAMID_LEVEL]; //points to depth variance
    
    Mat depth_pyramid[util::MAX_PYRAMID_LEVEL];
    
    Mat gradientx;
    
    Mat gradienty;
    
    Mat maxAbsGradient;
    
    Mat mask; //contains 1: to mark valid pixels which contain non-zero depth, 0: for zero depth
    
    int currentRows,currentCols;
    
    int pyrLevel;
    
    int no_nonZeroDepthPts; //num of non-zero depth pixels
    
    //Pose matrices
    float poseWrtOrigin[6]; //wrt KF
    
    float poseWrtWorld[6]; //w.r.t world, i.e first frame
    
    float truePoseWrtWorld[6];
    
    float rescaleFactor; //wrt KF
    
    Eigen::Matrix4f SE3poseThisWrtOther;
    Eigen::Matrix4f SE3poseOtherWrtThis;
    Eigen::Matrix3f SE3poseThisWrtOther_r;
    Eigen::MatrixXf SE3poseThisWrtOther_t;
    Eigen::Matrix3f SE3poseOtherWrtThis_r;
    Eigen::MatrixXf SE3poseOtherWrtThis_t;
    
    Eigen::Matrix4f K_SE3poseThisWrtOther;
    Eigen::Matrix4f K_SE3poseOtherWrtThis;
    Eigen::Matrix3f K_SE3poseThisWrtOther_r;
    Eigen::MatrixXf K_SE3poseThisWrtOther_t;
    Eigen::Matrix3f K_SE3poseOtherWrtThis_r;
    Eigen::MatrixXf K_SE3poseOtherWrtThis_t;

    
    Eigen::MatrixXf SE3_R;
    Eigen::MatrixXf Sim3_R; //not correct
    Eigen::MatrixXf SE3_T;
    Eigen::MatrixXf SE3_Pose;
    
    //member functions

    void constructImagePyramids();
    
    void calculateGradient();
    
    void initializeRandomly();
    
    void getImgSize(int& row,int& col,int level);
    
    void calculateNonZeroDepthPts();
    
    void initializePose();
    
    void updationOnPyrChange(int level,bool isPrevious=true);

    void calculatePoseWrtOrigin(frame* prev_image,float* poseChangeWrtPrevframe,bool frmhomo=false);
    
    void calculatePoseWrtWorld(frame* prev_image,float* poseChangeWrtPrevframe,bool frmhomo=false);
    
    void calculateSE3poseOtherWrtThis(frame *other_frame); //This To Other
    
    void calculateSim3poseOtherWrtThis( float scale_factor);

    void buildMaxGradients();

    void buildInvVarDepth(int level);
    
    void calculateRandT(void);
    
    void calculateRelativeRandT(float* relPose);
    
    void concatenateRelativePose(float* src_1wrt2, float* src_2wrt3, float* dest_1wrt3  );
    
    void concatenateOriginPose(float* src_1wrt0, float* src_2wrt0, float* dest_1wrt2 );
    
    void calculateInvLiePose(float* pose);
    
    void calculateInvLiePose(float *posesrc, float *posedest);
    
    void finaliseWeights();
    
    void saveMatAsText(Mat save_mat, string name, string save_mat_path);
    
    void makeMatFromText(Mat make_mat, string name, string read_txt_path);
    
    void saveArrayAsText(float* save_arr, string name, string save_arr_path, int pyr_level);
    
    void makeArrayFromText(float* make_arr, string name, string read_txt_path, int pyr_level);
    
    void makeDepthHypothesisFromText(depthhypothesis* make_depth_hypo, string name, string read_txt_path);

    //for interpolation of image intensity
    inline float getInterpolatedElement(float x1,float y1, int checkOutfBound=0){
        //PRINTF("\nCalulating Interpolated Image Intensity for frame Id: %d and points(%f,%f) ", frameId, x1, y1 );
        
        //initialize pointers
        uchar* img_ptr0; //pointer0 to access original image
        uchar* img_ptr1; //pointer1 to acess original image
        
        //*******DECLARE VARIABLES*******//
        
        float yx[2]; //to store warped points for current itertion
        float wt[2]; //to store weight
        float y,x;
        uchar pixVal1, pixVal2; //to store intensity value
        float interTop, interBtm;
        
        int nCols=currentCols-1; //maximum valid x coordinate
        int nRows=currentRows-1; //maximum valid y coordinate
        
        int countOutOfBounds=0;
        
        yx[0]=y1; //store current warped point y
        yx[1]=x1; //store warped point x
        
        wt[0]=yx[0]-floor(yx[0]); //weight for y
        wt[1]=yx[1]-floor(yx[1]); //weight for x
        
        //Case 1
        y=floor(yx[0]); //floor of y
        x=floor(yx[1]); //floor of x
        
        if ((x<0)||(x>nCols)||(y<0)||(y>nRows))
        {
            pixVal1=0; //if outside boundary pixel value is 0
            countOutOfBounds++;
        }
        else
        {
            img_ptr0=image_pyramid[pyrLevel].ptr<uchar>(y); //initialize image pointer0 to row floor(y)
            pixVal1=img_ptr0[int(x)]; //move pointer to get value at pixel (floor(x), floor(y))
        }
        
        x=yx[1]; //warped point x
        
        if ((x<0)||(x>nCols)||(y<0)||(y>nRows))
        {
            pixVal2=0; //if outside boundary pixel value is 0
            countOutOfBounds++;
        }
        else
        {
            img_ptr0=image_pyramid[pyrLevel].ptr<uchar>(y); //initialize image pointer0 to row floor(y)
            pixVal2=img_ptr0[int(ceil(x))]; //move pointer to get value at pixel (ceil(x), floor(y))
        }
        
        interTop=((1-wt[1])*pixVal1)+(wt[1]*pixVal2);
        
        //Case 2
        y=yx[0]; //warped point y
        
        x=floor(yx[1]); //floor of x
        
        if ((x<0)||(x>nCols)||(y<0)||(y>nRows))
        {
            pixVal1=0; //if outside boundary pixel value is 0
            countOutOfBounds++;
        }
        else
        {
            img_ptr1=image_pyramid[pyrLevel].ptr<uchar>(ceil(y)); //initialize image pointer1 to row ceil(y)
            pixVal1=img_ptr1[int(x)]; //move pointer to get value at pixel (floor(x), ceil(y))
        }
        
        x=yx[1]; //warped point x
        
        if ((x<0)||(x>nCols)||(y<0)||(y>nRows))
        {
            pixVal2=0; //if outside boundary pixel value is 0
            countOutOfBounds++;
        }
        else
        {
            img_ptr1=image_pyramid[pyrLevel].ptr<uchar>(ceil(y)); //initialize image pointer1 to row ceil(y)
            pixVal2=img_ptr1[int(ceil(x))]; //move pointer to get value at pixel (ceil(x), ceil(y))
        }
        
        float warpedvalue;
        if(countOutOfBounds==4 && checkOutfBound==1)
        {
            warpedvalue=-1.0f;
        }
        else
        {
            interBtm=((1-wt[1])*pixVal1)+(wt[1]*pixVal2);
            warpedvalue=((1-wt[0])*interTop)+(wt[0]*interBtm); //calculate interpolated value to get warped image intensity
        }
        
        return warpedvalue;
        
    }

  
    //for interpolation of gradient
    inline float getInterpolatedElement(float x1, float y1,String s)
    {
        if(isinf(x1) || isinf(y1))
            printf("\nInf Error in Get Interpolated: x1: %f, y1: %f", x1, y1);
            
        //PRINTF("\nCalculating Interpolated Gradient(x or y) Intensity for Frame Id: %d and points(%f,%f) ", frameId, x1, y1);
        //initialize pointers
        float* img_ptr0; //pointer0 to access original image
        float* img_ptr1; //pointer1 to acess original image
        
        Mat image1;
        if (s=="gradx") {
            image1=gradientx;
        }
        if (s=="grady") {
            image1=gradienty;
        }
        //*******DECLARE VARIABLES*******//
        
        //cout<<"\n\nWARPED POINTS   "<<warpedpoint;
        
        float yx[2]; //to store warped points for current itertion
        float wt[2]; //to store weight
        float y,x;
        float pixVal1, pixVal2; //to store intensity value
        float interTop, interBtm;
        
        int nCols=currentCols-1; //maximum valid x coordinate
        int nRows=currentRows-1; //maximum valid y coordinate
        
        
        yx[0]=y1; //store current warped point y
        yx[1]=x1; //store warped point x
        
        wt[0]=yx[0]-floor(yx[0]); //weight for y
        wt[1]=yx[1]-floor(yx[1]); //weight for x
        
        //Case 1
        y=floor(yx[0]); //floor of y
        x=floor(yx[1]); //floor of x
        
        //cout<<"\n\n"<<int(x);
        if ((x<0)||(x>nCols)||(y<0)||(y>nRows))
            pixVal1=0; //if outside boundary pixel value is 0
        else
        {
            img_ptr0=image1.ptr<float>(y); //initialize image pointer0 to row floor(y)
            pixVal1=img_ptr0[int(x)]; //move pointer to get value at pixel (floor(x), floor(y))
        }
        
        if(isnan(pixVal1))
            printf("\npix val1: %f, x1: %f, y1: %f, x: %f, y: %f", pixVal1, x1, y1, x, y);
        
        x=yx[1]; //warped point x
        
        if ((x<0)||(x>nCols)||(y<0)||(y>nRows))
            pixVal2=0; //if outside boundary pixel value is 0
        else
        {
            img_ptr0=image1.ptr<float>(y); //initialize image pointer0 to row floor(y)
            pixVal2=img_ptr0[int(ceil(x))]; //move pointer to get value at pixel (ceil(x), floor(y))
        }
        
        if(isnan(pixVal2))
            printf("\npix val2: %f, x1: %f, y1: %f, x: %f, y: %f", pixVal2, x1, y1, x, y);

        
        interTop=((1-wt[1])*pixVal1)+(wt[1]*pixVal2);
        
        
        //Case 2
        y=yx[0]; //warped point y
        
        x=floor(yx[1]); //floor of x
        
        if ((x<0)||(x>nCols)||(y<0)||(y>nRows))
            pixVal1=0; //if outside boundary pixel value is 0
        else
        {
            img_ptr1=image1.ptr<float>(ceil(y)); //initialize image pointer1 to row ceil(y)
            pixVal1=img_ptr1[int(x)]; //move pointer to get value at pixel (floor(x), ceil(y))
        }
        
        if(isnan(pixVal1))
            printf("\npix val1: %f, x1: %f, y1: %f, x: %f, y: %f", pixVal1, x1, y1, x, y);

        
        x=yx[1]; //warped point x
        
        if ((x<0)||(x>nCols)||(y<0)||(y>nRows))
            pixVal2=0; //if outside boundary pixel value is 0
        else
        {
            img_ptr1=image1.ptr<float>(ceil(y)); //initialize image pointer1 to row ceil(y)
            pixVal2=img_ptr1[int(ceil(x))]; //move pointer to get value at pixel (ceil(x), ceil(y))
        }
        
        if(isnan(pixVal2))
            printf("\npix val2: %f, x1: %f, y1: %f, x: %f, y: %f", pixVal2, x1, y1, x, y);

        
        interBtm=((1-wt[1])*pixVal1)+(wt[1]*pixVal2);
        
        float warpedvalue=((1-wt[0])*interTop)+(wt[0]*interBtm); //calculate interpolated value to get warped image intensity
        
        if(isnan(warpedvalue))
            printf("\nwarped val: %f, x1: %f, y1: %f, interTop: %f, interBtm: %f, wt0: %f, wt1: %f", warpedvalue, x1, y1, interTop, interBtm, wt[0], wt[1]);
        
        return warpedvalue;
        
        
    }

    
};


#endif /* defined(__Frame__) */
