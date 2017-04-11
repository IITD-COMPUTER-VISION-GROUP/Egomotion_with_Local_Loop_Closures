/**
 * This file is a part of Egomotion with Local Loop Closures(ELLC) project.
 * For more information visit: http://www.cse.iitd.ac.in/~suvam/wacv2017.html
 **/

#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <unsupported/Eigen/MatrixFunctions>

#include <cstdio>
#include <ctime>
#include <cstdlib>
#include "opencv2/opencv.hpp"
#include <complex>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>
#include <sstream>

#include "UserDefinedFunc.h"
#include "ExternVariable.h"
#include "DisplayFunc.h"
#include "Pyramid.h"
#include "PixelWisePyramid.h"


#include "ImageFunc.h"


using namespace std;
using namespace cv;

/*
this function estimates pose of current frame w.r.t keyframe or ref frame (here named as prev frame) using gauss newton
it can use either non-zero initialization or pose of tminus frame(i.e frame just prior to current frame) for initialization
this is based on the hypothesis that the pose between consecutive frames differs by a small value, therefore initialization this way can lead to faster convergence of GN.

this supports both pixel wise and non-pixel wise (i.e using entire matrix) pose estimation. these two are same except for that the former can be parallelized
this supports both forward compositional gauss newton algo(that re-estimates gradient and weights in each iteration, and is slower) and inverse compositional gauss newton algo (that fixes or precomputes weights and gradient, and is faster)
 
finally using the estimated pose of current frame w.r.t keyframe, it also updates its world pose (pose w.r.t frame 1) and origin pose (frame w.r.t kf) variables
 
flags passed in:
fromLoopClosure: which indicates whether this function is being called during loop closure for extra matches
hom: not used (was for homograohy)
 
*/
vector<float> GetImagePoseEstimate(frame* prev_frame, frame* current_frame, int frame_num,depthMap* currDepthMap,frame* tminus1_prev_frame,float* initial_pose_estimate,bool fromLoopClosure, bool homo)
{
    
    util::measureTime time_pose;
    time_pose.startTimeMeasure();
    
    //not used
    //only revives keyframe mats and arrays
    //save depth of prev frames (now refined, since new keyframe will be initialized)
    if(util::FLAG_REPLICATE_POSE_ESTIMATION && !fromLoopClosure && current_frame->frameId%util::KEYFRAME_PROPAGATE_INTERVAL==0)
    {
        //make depth mats from text file
        prev_frame->makeMatFromText(prev_frame->depth, "Depth", util::SAVED_MATS_PATH);
        //DisplayColouredDepth(prev_frame->depth, prev_frame->image);
        prev_frame->makeMatFromText(prev_frame->depth_pyramid[0], "Depth_pyr0", util::SAVED_MATS_PATH);
        //make depth var arrays from text file
        prev_frame->makeArrayFromText(currDepthMap->depthvararrpyr0, "DepthVarArr_pyr0", util::SAVED_MATS_PATH, 0);
    }
    
    
    
    //only saves keyframe mats and arrays at the highest level
    
    //when alternate_gn_ra is on then Flag_save_mats is On only for one keyframe prop
    //when alterna_gn_ra is off,then for less than 50
    //when alternate_gn_ra is on and bootstrap is on, then for less than 50
    if(util::FLAG_SAVE_MATS && !fromLoopClosure && (current_frame->frameId%util::KEYFRAME_PROPAGATE_INTERVAL==0) )
   
    {
        if((util::FLAG_ALTERNATE_GN_RA && util::NUM_GN_PROPAGATION<2 && util::GAUSS_NEWTON_FLAG_ON) || ((!util::FLAG_ALTERNATE_GN_RA) && (current_frame->frameId<50)) || ((util::FLAG_ALTERNATE_GN_RA) && (util::FLAG_IS_BOOTSTRAP) &&(current_frame->frameId<50)))
    {
        
        prev_frame->saveMatAsText(prev_frame->depth, "Depth", util::SAVED_MATS_PATH);
        prev_frame->saveMatAsText(prev_frame->depth_pyramid[0], "Depth_pyr0", util::SAVED_MATS_PATH);
        
        prev_frame->saveArrayAsText(currDepthMap->depthvararrpyr0, "DepthVarArr_pyr0", util::SAVED_MATS_PATH, 0);
 
    }
    }
    
    
    //PRINTF("\nCalculating Image Pose Estimate using current frame: %d and previous frame: %d", current_frame->frameId ,prev_frame->frameId);
    
    float pose[6];
    float initial_rel_pose[6];
   
    //initializing pose
    
    if(!util::FLAG_INITIALIZE_NONZERO_POSE || fromLoopClosure) //if initialization flag off, then initialized with pose of tminus1 frame
    {
        pose[0]=0.0f;
        pose[1]=0.0f;
        pose[2]=0.0f;
        pose[3]=0.0f;
        pose[4]=0.0f;
        pose[5]=0.0f;
        
        prev_frame->concatenateOriginPose(tminus1_prev_frame->poseWrtWorld, prev_frame->poseWrtWorld, pose);
        
    }
    else //if initialization flag on, then initial world pose is converted to origin pose (w.r.t kf)
    {
        pose[0]=initial_pose_estimate[0];
        pose[1]=initial_pose_estimate[1];
        pose[2]=initial_pose_estimate[2];
        pose[3]=initial_pose_estimate[3];
        pose[4]=initial_pose_estimate[4];
        pose[5]=initial_pose_estimate[5];
        
        //rotation part comes from given initialzed pose
        prev_frame->concatenateOriginPose(initial_pose_estimate, prev_frame->poseWrtWorld, pose);
    
        float pose_trans[6];
        prev_frame->concatenateOriginPose(tminus1_prev_frame->poseWrtWorld, prev_frame->poseWrtWorld, pose_trans);
        
        //translation part comes from tminus1 frame
        //another option is to initialzie translation as 0
        pose[3]=pose_trans[3];
        pose[4]=pose_trans[4];
        pose[5]=pose_trans[5];

        
        initial_rel_pose[0]=pose[0];
        initial_rel_pose[1]=pose[1];
        initial_rel_pose[2]=pose[2];
        initial_rel_pose[3]=pose[3];
        initial_rel_pose[4]=pose[4];
        initial_rel_pose[5]=pose[5];

    }

    //initializing flags for image display
    int display_initial_img_flag=0;
    int display_orig_img_flag=0;
    
    int iter_counter;
    
    
    
  //*******ENTER PYRAMID LOOP*******//

for (int level_counter=(util::MAX_PYRAMID_LEVEL-1); level_counter>=0; level_counter--)
{
    
    //*******PRECOMPUTATION BEGINS*******//
    
    //printf("\nPyramid level: %d",level_counter);
    
    //updates pyramid dependent farme parameters
    prev_frame->updationOnPyrChange(level_counter);
    current_frame->updationOnPyrChange(level_counter,false);

    
    //2 instances created here, however, only 1 used at a time
    Pyramid workingPyramid(prev_frame, current_frame, pose,currDepthMap); // normal working pyramid
    PixelWisePyramid workingPixelPyramid(prev_frame, current_frame, pose,currDepthMap); //pixel-wise working pyramid that can be parallelized
    
    
    //initialization!
    //CASE 1: when non-parallel pose est. OR when normal (i.e. non-constant weight) loop closure pose est.
    if(!util::FLAG_DO_PARALLEL_POSE_ESTIMATION || (!util::FLAG_DO_CONST_WEIGHT_POSE_ESTIMATION && fromLoopClosure))
    {
        workingPyramid.putPreviousPose(tminus1_prev_frame);
        workingPyramid.pose=pose;
    
    //Perform precomputation--Calculation of steepest descent,hessian inverse & world points.
    workingPyramid.performPrecomputation();
        
    }
    
    //CASE 2: when parallel pose est OR when constant weight loop closure pose est.
    if(util::FLAG_DO_PARALLEL_POSE_ESTIMATION || (util::FLAG_DO_CONST_WEIGHT_POSE_ESTIMATION && fromLoopClosure))
    {
        workingPixelPyramid.putPreviousPose(tminus1_prev_frame);
        workingPixelPyramid.pose=pose;
    }

    display_orig_img_flag=0; //reset flag
    int disp_iter_counter=0;
    
    
    //*******ENTER ITERATION LOOP*******//
    
    for(iter_counter=0; iter_counter<util::MAX_ITER[level_counter];  ++iter_counter)
    {
        disp_iter_counter=iter_counter;

        //printf("\nIteration: %d ",iter_counter);
        
        //Perform iteration steps-- Calculation of warped points,warped image & residual. Also updates pose and checks condition for loop termination.

        
        //CASE 1: when non-parallel pose est. OR when normal (i.e. non-constant weight) loop closure pose est.
        if(!util::FLAG_DO_PARALLEL_POSE_ESTIMATION || (!util::FLAG_DO_CONST_WEIGHT_POSE_ESTIMATION && fromLoopClosure))
        {
            if(workingPyramid.weightedPose<1.0f) //if true, pose change insignificant, therefore terminate iteration at this level
                iter_counter=util::MAX_ITER[level_counter]-1;
            
            //to display current iteration residual
            if(util::FLAG_DISPLAY_IMAGES)
            {
                DisplayIterationRes(prev_frame, workingPyramid.residual, "Iteration Residual", frame_num,homo);
                DisplayWeights(prev_frame, workingPyramid.weights, "Weight Image", frame_num, homo);
                
                //to display original image residual and original image
                if(display_initial_img_flag==0)
                {
                    //display initial residual
                    DisplayInitialRes(current_frame,prev_frame,"Initial Residual", frame_num,homo);
                    
                    display_initial_img_flag=1; //set flag
                }
                //diplay original image
                if(display_orig_img_flag==0)
                {
                    DisplayOriginalImg(workingPyramid.saveImg,prev_frame,"Original Image", frame_num,homo);
                    display_orig_img_flag=1; //set flag
                }

                //to display warped image
                DisplayWarpedImg(workingPyramid.warpedImage, prev_frame, "Warped Image", frame_num,homo);
            }
            
            
        }
        
        //CASE 2: when parallel pose est OR when constant weight loop closure pose est.
        if(util::FLAG_DO_PARALLEL_POSE_ESTIMATION || (util::FLAG_DO_CONST_WEIGHT_POSE_ESTIMATION && fromLoopClosure)) //for pixel-wise
        {
            //printf("\nDo parallel pose: %d , do const weight: %d, from loop closure: %d, iter: %d", util::FLAG_DO_PARALLEL_POSE_ESTIMATION, util::FLAG_DO_CONST_WEIGHT_POSE_ESTIMATION, fromLoopClosure, iter_counter);
           
            //if using constant weight, then perform inverse compositional GN
            if(fromLoopClosure && util::FLAG_DO_CONST_WEIGHT_POSE_ESTIMATION)
            {
                workingPixelPyramid.calculatePixelWiseParallelInvCompositional(iter_counter); //when from loop closure and const weight flag set
            }
            else //else the normal pixel-wise GN
            {
               workingPixelPyramid.calculatePixelWiseParallel(); //when not from loop closure or const weight flag not set
            }
            
            //check termination
            if(workingPixelPyramid.weightedPose<1.0f) //if true, pose change insignificant, therefore terminate iteration at this level
                iter_counter=util::MAX_ITER[level_counter]-1;
            
            //to display current iteration residual
            if(util::FLAG_DISPLAY_IMAGES)
            {
                DisplayIterationResPixelWise(workingPixelPyramid.display_iterationres, prev_frame,"Iteration Residual",frame_num,homo);
                DisplayWeightsPixelWise(workingPixelPyramid.display_weightimg, prev_frame, "Weight Image", frame_num, homo);

                //to display original image residual and original image
                if(display_initial_img_flag==0)
                {
                    //display initial residual
                    DisplayInitialRes(current_frame,prev_frame,"Initial Residual", frame_num,homo);
                       waitKey(1000);
                    display_initial_img_flag=1; //set flag
                }
                //diplay original image
                if(display_orig_img_flag==0)
                {
                    DisplayOriginalImgPixelWise(workingPixelPyramid.prev_frame->image_pyramid[level_counter], prev_frame, "Original Image", frame_num,homo);
                    display_orig_img_flag=1; //set flag

                }
                
                //to display warped image
                DisplayWarpedImgPxelWise(workingPixelPyramid.display_warpedimg, prev_frame, "Warped Image", frame_num,homo);

            }
            if(util::FLAG_DO_CONST_WEIGHT_POSE_ESTIMATION && !fromLoopClosure)
            {
                //save final weights at end of every level
                if(iter_counter==util::MAX_ITER[level_counter]-1)
                {
                    workingPixelPyramid.saveWeights(true);
                    //DisplayWeightsPixelWise(current_frame->weights, prev_frame, "Saved Weight Image", frame_num);
                }
            }
        }
        
   
    } //exit iteration loop
    

    //PRINTF("\nSummary: For previous frameId: %d, current frameId: %d ", prev_frame->frameId, current_frame->frameId);
   // printf("\nAt Pyramid Level: %d , Max Iteration: %d ",level_counter,disp_iter_counter+1);

    
} //exit pyramid loop

    
    //PRINTF("\nUpdated pose: %f ,%f , %f , %f , %f , %f ",pose[0],pose[1],pose[2],pose[3],pose[4],pose[5]);
    
    //update pose variables of current farme
    current_frame->calculatePoseWrtOrigin(prev_frame,pose); //saves pose w.r.t kf
    current_frame->calculatePoseWrtWorld(prev_frame,pose); //saves pose w.r.t world
    current_frame->calculateRandT(); //extracts rot matrix and ranslation from pose
    
 
    //PRINTF("\nExiting frame loop..");
    vector<float> posevec(pose, pose+6);
    
    return posevec; //returns relative pose between current frame and kf
    
}

