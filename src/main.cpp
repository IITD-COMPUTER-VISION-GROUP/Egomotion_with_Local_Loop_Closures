/**
 * This file is a part of Egomotion with Local Loop Closures(ELLC) project.
 * For more information visit: http://www.cse.iitd.ac.in/~suvam/wacv2017.html
 **/

#include <cstdio>
#include <ctime>
#include <cstdlib>
#include "opencv2/opencv.hpp"
#include <complex>
#include <string>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>

#include "DepthPropagation.h"
#include "DisplayFunc.h"
#include "ImageFunc.h"
#include "ExternVariable.h"
#include "Frame.h"
#include "EigenInitialization.h"
#include "GlobalOptimize.h"
#include "ToggleFlags.h"

using namespace std;
using namespace cv;

//---------------------------------------------

// EXTERN VARS INIIALIZATION
// These are non-constant flags/variables that can be changed by program during alternate RA-GN mode

int util::MAX_ITER[]={4,7,9,12} ;           //maximum GN iteration at different levels, 0->highest image size, 3->smallest image size

// Alternate GN, RA

bool util::FLAG_ALTERNATE_GN_RA = false; //default mode: GAUSS_NEWTON W/O ROT. AVG.;
// These are used in alternate GN-RA mode
int  util::FLAG_IS_BOOTSTRAP=false;         //read from config file
int  util::BATCH_START_ID=0;                //read from config file
int  util::BATCH_SIZE=0;                    //read from config file, make default iniialization as 1!!! (in terms of number of keyframe propagations needed)
int  util::NUM_GN_PROPAGATION=0;
int  util::NUM_RA_PROPAGATION=0;
bool util::GAUSS_NEWTON_FLAG_ON=false;
bool util::ROTATION_AVERAGING_FLAG_ON=false;

// Toggle flags depending on requirement
bool util::FLAG_DO_LOOP_CLOSURE=false;         //(LC) to find extra poses through short loop closure detection
bool util::FLAG_REPLICATE_NEW_DEPTH=false;              //use a saved depth matrix saved as a text file during pose est.
bool util::FLAG_INITIALIZE_NONZERO_POSE=false;          //use non-zero pose as initialization
bool util::FLAG_SAVE_MATS=false;                        //save depth matrices as text files
bool util::FLAG_DO_PARALLEL_SHORT_LOOP_CLOSURE=false;   //(LC) use multi-threads for loop closure

//flags within loop closure
bool util::FLAG_DO_CONST_WEIGHT_POSE_ESTIMATION=false; //(LC) use constant weights during LC for extra matches
bool util::FLAG_DO_PARALLEL_CONST_WEIGHT_POSE_EST=false; //(LC) use multi-threaded implementation of const. weight
bool util::FLAG_USE_LOOP_CLOSURE_TRIGGER=false; //default value, dont change

bool util::EXIT_CONDITION=false; //default value, don't change

//---------------------------------------------


// Fstream objects to read/write
ifstream my_file;               //for config.txt
ofstream pose_file_orig;        //to write orig poses
ofstream match_file;            //to write extra matches
ifstream initialize_pose_file;  //read non-zero pose initializations


// Give path to config.txt as command line argument
int main(int argc, char *argv[])
{
 
    if(argc==1)
    {
         printf("Flag for Local Loop Closure not provided. Switching to DEFAULT mode: GAUSS_NEWTON W/O ROT.AVG.\n");
    }
    
    if(argc==2)
    {
        printf("\nEither Config. file or loop closure flag missing! Exiting...\n");
        return -1;
    }
    
    
    
    if(argc==3)
    {
        if(!strcmp(argv[1],"LC"))
        util::FLAG_ALTERNATE_GN_RA = true;
    
    
    my_file.open(argv[2]);
    
    if (!my_file.is_open())                     // check if we succeeded with config. file
    {
        printf("\nUnable to open Config. file! Exiting...\n");
        return -1;
    }
    }
    
    vector<frame*> frameptr_vector;             // vector of pointers to each frame currently in memory
    
    //*******READING INPUT FILE*******//
    string vid_path;
    vid_path = "../data/%06d.jpg";                            //path of video/images
    
    string match_path;                          //second line: path to write relative poses between original rame and its keyframe
    match_path = "../outputs/matchframes.txt";
    match_file.open(match_path);
    
    string match_path_globalopt;                //third line: path to write relative poses of extra matches calculated through global optimization
    match_path_globalopt = "../outputs/matchframes_globalopt.txt";
    
    string depthmap_path;                       //fourth line: path of folder to store depth map associated with kf
    depthmap_path = "../Test_images/";
    
    string pose_path_orig;                      //fifth line: path to write absolute (w.r.t world) poses of each frame computed on its original kf
    pose_path_orig = "../outputs/poses_orig.txt";
    pose_file_orig.open(pose_path_orig);
    
    string matchsave_path;                      //sixth line: path to folder to save matched images on which extra matches are computed
    matchsave_path = "../matches/";
    
    string initialize_pose_path;                //seventh line: path to text file containing absolute(w.r.t world) poses to be used as initializations
    initialize_pose_path = "../outputs/so3poses7.txt";
    initialize_pose_file.open(initialize_pose_path);
    
    //set all batch parameters
    if(util::FLAG_ALTERNATE_GN_RA)
    {
        my_file>>util::BATCH_START_ID;          //set start frame id of batch
        my_file>>util::BATCH_SIZE;            //set batch size in terms of kf propagations
        my_file>>util::FLAG_IS_BOOTSTRAP;
    }

    bool toExit=false;

    if(util::FLAG_ALTERNATE_GN_RA )
    {
        toExit=checkExitCondition();            //updates GN and RA flags
    }
   
    //*******VIDEO INITIALIZATIONS*******//
    VideoCapture bgr_capture(vid_path);
    
    if (!bgr_capture.isOpened())                // check if we succeeded with BGR
    { 
        printf("\nUnable to open Frame File!");
        return -1;
    }
    
    //skips all images in folder till first frame of batch is reached
    if(util::FLAG_REPLICATE_NEW_DEPTH || (util::FLAG_IS_BOOTSTRAP && (util::BATCH_START_ID>0)))
    {
        Mat temp_im;
        int m;
        for(m=1;m<util::BATCH_START_ID;m++)
        {
            bgr_capture>>temp_im;
            cout<<"\nSkipping frame: "<<m;
            waitKey(5);
        }
    }
    
    
    //*******MAIN PARAMETER INITIALIZATIONS*******//
    
    //get no of frames in video
    float no_of_frames = bgr_capture.get(CV_CAP_PROP_FRAME_COUNT);      //get the number of frames in depth
    
    if(no_of_frames>32500)                       //cannot be greater than max value stored in int
        no_of_frames=32500;
    
    vector<float>pose_vec;
    
    printf("\nNo of frames:%f",no_of_frames);
    
    frame* activeKeyFrame=NULL;
    
    
    depthMap* currentDepthMap=new depthMap();
    
    currentDepthMap->save_depthmap_path=depthmap_path;
    
    
    globalOptimize globalOptimizeLoop(match_path_globalopt); //does global optimizations (i.e over short loop closures)
    
    int num_keyframe=0;
    float seeds_num;
    int max_frame_counter=no_of_frames;                      //maximum frames to run pose est.
    
    
    //*******ENTER FRAME LOOP*******//
    float new_world[6]={0,0,0,0,0,0};
    int frame_counter;
    for (frame_counter = 1; frame_counter <= max_frame_counter; frame_counter++)
    {
        
    
        frameptr_vector.push_back(new frame(bgr_capture));  //push input frame in vector
        
        float initial_pose[6];
        
        if(util::FLAG_INITIALIZE_NONZERO_POSE)              //if flag set, initializes pose to be used in GN from text file
        {
            int temp_frame_no;
            initialize_pose_file>>temp_frame_no>>initial_pose[0]>>initial_pose[1]>>initial_pose[2]>>initial_pose[3]>>initial_pose[4]>>initial_pose[5];
            
            if(util::FLAG_CONCATENATE_INITIAL_POSE)
            {
                    float concatenated_initial_pose[6]={0,0,0,0,0,0};
                    frameptr_vector.back()->concatenateRelativePose(initial_pose, new_world, concatenated_initial_pose);
                    
                    //copy only concatenated rotations to initial pose
                    initial_pose[0]=concatenated_initial_pose[0];
                    initial_pose[1]=concatenated_initial_pose[1];
                    initial_pose[2]=concatenated_initial_pose[2];
            }
            
            
            //printf("\n\nINITIAL ABS pose: %f, %f, %f, %f, %f, %f", initial_pose[0], initial_pose[1], initial_pose[2], initial_pose[3], initial_pose[4], initial_pose[5]);
        }
    
        // To bootstrap the first image in the sequence
        if(frame_counter==1)
        {
            activeKeyFrame=frameptr_vector.back();                  //first active keyframe is first frame
            num_keyframe++;
            currentDepthMap->formDepthMap(frameptr_vector.back());  //does random initialization
            currentDepthMap->updateDepthImage();
            
             continue;
        }
        
        //not used
        //to re-estimate only at keyframes, since relative poses between frame and kf will remain unchanged
        if(util::FLAG_REPLICATE_POSE_ESTIMATION)
        {
            if(frame_counter%util::KEYFRAME_PROPAGATE_INTERVAL!=0)
                continue;
        }
        
        //connection recovery mechanism which finds new depth map from history in the event that current depth has few/no seeds left
        //checks each incoming frame with history of frames
        //if match found then tries to estimate pose using saved depth map and then propagates depth map to check if seeds are abive threshold
        //if no match found, then drops this frame
        
        //check if connection present between keyframe and current frame
        if(util::FLAG_RESTORE_CONNECTION) //set flag to use this
        {    
            globalOptimizeLoop.checkConnection(currentDepthMap);
        
            //find new connection if connection is lost
            if(globalOptimizeLoop.connectionLost==true)
            {            
                //searches loop array for new connection
                globalOptimizeLoop.findConnection(frameptr_vector.back());
                
                if(globalOptimizeLoop.connectionLost==false)    //new connection found!
                {
                    printf("\nConnection has been FOUND! Updating variables!");
                    delete currentDepthMap;                     //delete existing current depth map
                
                    currentDepthMap=new depthMap(*globalOptimizeLoop.temp_depthMap); //re-initialize to new depth map
                    
                    //update pointers, not sure if this is needed
                    currentDepthMap->depthvararrptr[0]=currentDepthMap->depthvararrpyr0;
                    currentDepthMap->depthvararrptr[1]=currentDepthMap->depthvararrpyr1;
                    currentDepthMap->depthvararrptr[2]=currentDepthMap->depthvararrpyr2;
                    currentDepthMap->depthvararrptr[3]=currentDepthMap->depthvararrpyr3;
                    
                    currentDepthMap->deptharrptr[0]=currentDepthMap->deptharrpyr0;
                    currentDepthMap->deptharrptr[1]=currentDepthMap->deptharrpyr1;
                    currentDepthMap->deptharrptr[2]=currentDepthMap->deptharrpyr2;
                    currentDepthMap->deptharrptr[3]=currentDepthMap->deptharrpyr3;
                    
                    
                    seeds_num= currentDepthMap->calculate_no_of_Seeds();
                    
                    //writing new pose
                    if(util::FLAG_WRITE_ORIG_POSES)
                    {
                        //writing orig poses for normal keyframe sequence
                        if (pose_file_orig.is_open())
                        {
                            pose_file_orig<<(frameptr_vector.back()->frameId+util::BATCH_START_ID-1)<<" "<<(activeKeyFrame->frameId+util::BATCH_START_ID-1)<<" "<<frameptr_vector.back()->poseWrtWorld[0]<<" "<<frameptr_vector.back()->poseWrtWorld[1]<<" "<<frameptr_vector.back()->poseWrtWorld[2]<<" "<<frameptr_vector.back()->poseWrtWorld[3]<<" "<<frameptr_vector.back()->poseWrtWorld[4]<<" "<<frameptr_vector.back()->poseWrtWorld[5]<<" "<<activeKeyFrame->rescaleFactor<<" "<<seeds_num<<"\n";
                        }
                    }
                    
                    if(util::FLAG_WRITE_MATCH_POSES)
                    {
                        //only keyframes
                        if(frameptr_vector.back()->frameId%util::KEYFRAME_PROPAGATE_INTERVAL==0)
                        {
                            //pose wrt origin ->frame id, kf id                  
                            if(match_file.is_open())
                            {
                                match_file<<(frameptr_vector.back()->frameId+util::BATCH_START_ID-1)<<" "<<(activeKeyFrame->frameId+util::BATCH_START_ID-1)<<" "<<frameptr_vector.back()->poseWrtOrigin[0]<<" "<<frameptr_vector.back()->poseWrtOrigin[1]<<" "<<frameptr_vector.back()->poseWrtOrigin[2]<<" "<<frameptr_vector.back()->poseWrtOrigin[3]<<" "<<frameptr_vector.back()->poseWrtOrigin[4]<<" "<<frameptr_vector.back()->poseWrtOrigin[5]<<" "<<activeKeyFrame->rescaleFactor<<" "<<seeds_num<<" "<<"0"<<" "<<"0"<<" "<<"0"<<"\n";
                            }
                        }
                    }
                    
                    //updating keyframe pointers
                    currentDepthMap->keyFrame=frameptr_vector.back(); //change keyframe
                    activeKeyFrame=frameptr_vector.back();
                    
                    delete globalOptimizeLoop.temp_depthMap;
                    globalOptimizeLoop.temp_depthMap=NULL;
                    
                    printf("\nAsking for new frame");
                    continue; //ask for new frame to be mapped on newly created keyframe
                }
            
                //if no new connection has been found, go to next frame
                printf("\nConection still Lost");
                delete frameptr_vector.back(); //delete this frame, no use
                frameptr_vector.erase(frameptr_vector.begin()+frameptr_vector.size()-1);
                printf("\nAsking for new frame");
                continue;
            }
        }
        

       //control reaches here only when flag_restore_connection is off OR new connection has been found, current depth map propagated to non-zero seeds and active keyframe updated
       
       //function that estimates pose of current frame w.r.t active keyframe
       pose_vec=GetImagePoseEstimate(activeKeyFrame, frameptr_vector.back(), frame_counter, currentDepthMap, frameptr_vector[frameptr_vector.size()-2], initial_pose);
        
        //not useful
        if(util::FLAG_CONCATENATE_INITIAL_POSE)
        {
            if(frame_counter%util::CONCATENATE_STEP==0)
            {
                printf("\nUpdating New World!");
                //re-initialize with new world pose
                new_world[0]=frameptr_vector.back()->poseWrtWorld[0];
                new_world[1]=frameptr_vector.back()->poseWrtWorld[1];
                new_world[2]=frameptr_vector.back()->poseWrtWorld[2];
                new_world[3]=frameptr_vector.back()->poseWrtWorld[3];
                new_world[4]=frameptr_vector.back()->poseWrtWorld[4];
                new_world[5]=frameptr_vector.back()->poseWrtWorld[5];
                
            }
        }
        
        

        //rough calculation of newly estimated pose from initial non-zero pose
        //not sure if this can be done in lie algebra domain
        if(util::FLAG_INITIALIZE_NONZERO_POSE)
        {
            printf("\n\nFINAL Perc Change: %f, %f, %f, %f, %f, %f", (abs(initial_pose[0]-frameptr_vector.back()->poseWrtWorld[0])/abs(initial_pose[0]))*100, (abs(initial_pose[1]-frameptr_vector.back()->poseWrtWorld[1])/abs(initial_pose[1]))*100, (abs(initial_pose[2]-frameptr_vector.back()->poseWrtWorld[2])/abs(initial_pose[1]))*100, (abs(initial_pose[3]-frameptr_vector.back()->poseWrtWorld[3])/abs(initial_pose[3]))*100, (abs(initial_pose[4]-frameptr_vector.back()->poseWrtWorld[4])/abs(initial_pose[4]))*100, (abs(initial_pose[5]-frameptr_vector.back()->poseWrtWorld[5])/abs(initial_pose[5]))*100);
        }               
        


        //print poses
        seeds_num= currentDepthMap->calculate_no_of_Seeds();
        cout<<"\n"<<(frameptr_vector.back()->frameId+util::BATCH_START_ID-1)<<" "<<(activeKeyFrame->frameId+util::BATCH_START_ID-1)<<" "<<frameptr_vector.back()->poseWrtWorld[0]<<" "<<frameptr_vector.back()->poseWrtWorld[1]<<" "<<frameptr_vector.back()->poseWrtWorld[2]<<" "<<frameptr_vector.back()->poseWrtWorld[3]<<" "<<frameptr_vector.back()->poseWrtWorld[4]<<" "<<frameptr_vector.back()->poseWrtWorld[5]<<" "<<activeKeyFrame->rescaleFactor<<" "<<seeds_num;
        
        

        
        //writing newly calculated poses
        if(util::FLAG_WRITE_ORIG_POSES)
        {
            //writing orig poses (w.r.t world) for normal keyframe sequence
            if (pose_file_orig.is_open())
            {
                pose_file_orig<<(frameptr_vector.back()->frameId+util::BATCH_START_ID-1)<<" "<<(activeKeyFrame->frameId+util::BATCH_START_ID-1)<<" "<<frameptr_vector.back()->poseWrtWorld[0]<<" "<<frameptr_vector.back()->poseWrtWorld[1]<<" "<<frameptr_vector.back()->poseWrtWorld[2]<<" "<<frameptr_vector.back()->poseWrtWorld[3]<<" "<<frameptr_vector.back()->poseWrtWorld[4]<<" "<<frameptr_vector.back()->poseWrtWorld[5]<<" "<<activeKeyFrame->rescaleFactor<<" "<<seeds_num<<"\n";                
            }
        }
    
        if(util::FLAG_WRITE_MATCH_POSES)
        {
            //pose wrt origin (i.e. kf) ->frame id, kf id        
            if(match_file.is_open())
            {
                match_file<<(frameptr_vector.back()->frameId+util::BATCH_START_ID-1)<<" "<<(activeKeyFrame->frameId+util::BATCH_START_ID-1)<<" "<<frameptr_vector.back()->poseWrtOrigin[0]<<" "<<frameptr_vector.back()->poseWrtOrigin[1]<<" "<<frameptr_vector.back()->poseWrtOrigin[2]<<" "<<frameptr_vector.back()->poseWrtOrigin[3]<<" "<<frameptr_vector.back()->poseWrtOrigin[4]<<" "<<frameptr_vector.back()->poseWrtOrigin[5]<<" "<<activeKeyFrame->rescaleFactor<<" "<<seeds_num<<" "<<"0"<<" "<<"0"<<" "<<"0"<<"\n";
            }
        }
        

        //here pose estimation complete
        //now moving to depth propagation/refinement
        
       PRINTF("\n\n\nDEPTH MAP:");
       currentDepthMap->formDepthMap(frameptr_vector.back()); //updates depth map variables
        
        //not used
        if((frame_counter%util::KEYFRAME_PROPAGATE_INTERVAL!=0))
        {   
            if( util::FLAG_REPLICATE_POSE_ESTIMATION)
            {
                continue;
            }
        }
        

        // Checking for keyframe propagation
        if ( (frame_counter%util::KEYFRAME_PROPAGATE_INTERVAL==0) || (frame_counter==max_frame_counter))
        {
            if(util::FLAG_ALTERNATE_GN_RA)
            {
               updatePropagationCount();
            }
            
            //not used
            if(util::FLAG_REPLICATE_POSE_ESTIMATION)
            {
                //update active keyframes
                activeKeyFrame=frameptr_vector.back();
                currentDepthMap->keyFrame=activeKeyFrame;
                
                //delete all frames except most recent
                unsigned long size=frameptr_vector.size();
                unsigned long  i;
                for (i = 0; i < size-1; ++i)
                {
                    delete frameptr_vector[i]; // Calls ~object and deallocates *tmp[i]
                }
                frameptr_vector.erase(frameptr_vector.begin(),frameptr_vector.begin()+size-1);                
                
                continue;
            }
            
            
            if(util::FLAG_DO_CONST_WEIGHT_POSE_ESTIMATION) //save weights of keyframe to be used in extra match pose est.
            {
                activeKeyFrame->finaliseWeights();
            }
            
            currentDepthMap->finaliseKeyframe(); //finalises current depth map associated with active keyframe
            
            
            //checks exit condition (called from main)
            if(util::FLAG_ALTERNATE_GN_RA)
            {
                printf("\nChecking exit condition from Main...");
                toExit=checkExitCondition();
                if(toExit)
                {   
                    printf("\nDetected Exit condition in Main");
                    util::EXIT_CONDITION=true;
                   // break;
                }
            }
            
            if(util::FLAG_DO_LOOP_CLOSURE)
            {
                if(globalOptimizeLoop.connectionLost==true)
                {
                    globalOptimizeLoop.findConnection(frameptr_vector.back());
                }
                else
                {
                    //pushing active keyframe
                    //within this, it detects loop closure and find redundant poses
                    globalOptimizeLoop.pushToArray(activeKeyFrame, currentDepthMap); //pushes active keyframe
                
                }
            }
            
            if(util::FLAG_ALTERNATE_GN_RA && util::EXIT_CONDITION)
            {
                if(globalOptimizeLoop.loopClosureTerminated)
                    printf("\nLoop Closure Terminated. Returned in main.");
                else
                    printf("\nExit detected Again in Main");
                break;
            }
            
            //all computations on active keyframe has been completed, so now make new active keyframe and propagate depth map
            currentDepthMap->createKeyFrame(frameptr_vector.back());
            activeKeyFrame=frameptr_vector.back();
            
            
            
            //delete all frames except most recent
            unsigned long size=frameptr_vector.size();
            unsigned long i;
            for (i = 0; i < size-1; ++i)
            {
                delete frameptr_vector[i]; // Calls ~object and deallocates *tmp[i]
            }
            frameptr_vector.erase(frameptr_vector.begin(),frameptr_vector.begin()+size-1);
            

            printf("\nKEYFRAME TIME! Current Frame no: %d, do parallel depth: %d , do parallel pose: %d\n", frame_counter, util::FLAG_DO_PARALLEL_DEPTH_ESTIMATION, util::FLAG_DO_PARALLEL_POSE_ESTIMATION);
            
            continue;
        }
        
        //control reaches here when there is no keyframe propagation
        //does depth refinement and then updates associated depth matrix of active keyframe with newly refined depth
        currentDepthMap->updateKeyFrame();
        currentDepthMap->observeDepthRowParallel();
        currentDepthMap->doRegularization();
        currentDepthMap->updateDepthImage();
        
        
    } //exit frame loop
    
    
    return 0;
} //exit main

