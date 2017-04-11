/**
 * This file is a part of Egomotion with Local Loop Closures(ELLC) project.
 * For more information visit: http://www.cse.iitd.ac.in/~suvam/wacv2017.html
 **/

#pragma once

#ifndef __DepthPropagation__
#define __DepthPropagation__

#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include <vector>
#include <boost/thread/thread.hpp>
#include <boost/date_time.hpp>
#include <pthread.h>
#include <time.h>
#include <sys/time.h>

#include "DepthHypothesis.h"
#include "ExternVariable.h"
#include "Frame.h"



using namespace std;
using namespace cv;

/*
    This class contains the functions and variables associated with the current depth map
    The current depth map is the current estimate of depth, variance along with their regularized values for each pixel in the current keyframe 
 This class is partially adopted from LSD-SLAM
*/


class depthMap
{
    
    public:
    

    depthhypothesis currentDepthHypothesis[util::ORIG_COLS*util::ORIG_ROWS]; //array of depth hyposthesis, size is equal to img size\
    
    depthhypothesis otherDepthHypothesis[util::ORIG_COLS*util::ORIG_ROWS]; //used as a temporary array for swapping
    
    int validityIntegralBuffer[util::ORIG_COLS*util::ORIG_ROWS];
    
    int size;
    
    float depthScale;
    
    static int depthHypothesisId;
    
    frame* keyFrame; //points to reference frame for the current image, the depth hypothesis is aligned with the kf
    
    frame* currentFrame; //points to the current frame
    
    frame* activeDepthFrame; //points to the Origin for the Depth Map, not used

    //member functions
    depthMap(); //default constructor
    
    //hard-coded here for 4 pyramid levels, contains depth
    float deptharrpyr0[util::ORIG_COLS*util::ORIG_ROWS];
    float deptharrpyr1[(util::ORIG_ROWS>>1)*(util::ORIG_COLS>>1)];
    float deptharrpyr2[(util::ORIG_ROWS>>2)*(util::ORIG_COLS>>2)];
    float deptharrpyr3[(util::ORIG_ROWS>>3)*(util::ORIG_COLS>>3)];

    float* deptharrptr[util::MAX_PYRAMID_LEVEL]; //holds pointer to deptharrpyr
    
    //hard-coded here for 4 pyramid levels, contains varaince
    float depthvararrpyr0[util::ORIG_COLS*util::ORIG_ROWS];
    float depthvararrpyr1[(util::ORIG_ROWS>>1)*(util::ORIG_COLS>>1)];
    float depthvararrpyr2[(util::ORIG_ROWS>>2)*(util::ORIG_COLS>>2)];
    float depthvararrpyr3[(util::ORIG_ROWS>>3)*(util::ORIG_COLS>>3)];
    
    float* depthvararrptr[util::MAX_PYRAMID_LEVEL]; //holds pointer to depthvararrpyr
    
    string save_depthmap_path; //path to save the depth map

    void formDepthMap(frame* image_frame); //initialize parameters
    
    void initializeRandomly(); 
    
    bool makeAndCheckEPL(const int x, const int y, float* pepx, float* pepy);
    
    int observeDepthCreate(const int &x, const int &y, const int &idx);
    
    int observeDepthUpdate(const int &x, const int &y, const int &idx);

    void observeDepthRow(int ymin,int ymax, int thread_num);
    
    float doLineStereo( const float u, const float v, const float epxn, const float epyn, const float min_idepth, const float prior_idepth, float max_idepth, float &result_idepth, float &result_var, float &result_eplLength);
    
    void initializeDepthMapParameters(vector<frame*> frameptrvector_ptr);
    
    void propagateDepth(frame* new_keyframe); //propagates current depth map to depth of destination image
    
    void resetHypothesis(int idx);
    
    void displayColourDepthMap(frame* image_frame, bool fromKeyFrameCreation=false);
    
    void updateDepthImage(bool fromKeyFrameCreation= false);
    
    void fillDepthHoles();
    
    void regularizeDepthMap(bool removeOcclusions = false);
    
    void buildValIntegralBuffer();
    
    void makeInvDepthOne(bool calculate_on_current=true);
    
    void doRegularization(bool removeOcclusions = false);
    
    void buildInvVarDepth();
    
    void mapDepthArr2Mat();
    
    void finaliseKeyframe();

    void createKeyFrame(frame* new_keyframe);
    
    void updateKeyFrame();
    
    float calculate_no_of_Seeds(bool calculate_on_current=true);
    
    void checkHighVariancePatch();
    
    void observeDepthRowParallel();
    
    void scaleDepthMap();
    
};

#endif /* defined(__DepthPropagation__) */
