/**
 * This file is a part of Egomotion with Local Loop Closures(ELLC) project.
 * For more information visit: http://www.cse.iitd.ac.in/~suvam/wacv2017.html
 **/

#ifndef __ToggleFlags__
#define __ToggleFlags__

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <unsupported/Eigen/MatrixFunctions>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <time.h>
#include <sys/time.h>
#include <cmath>

#include "ExternVariable.h"



//uses save depth map and initialization of pose
inline void replicateNewDepthFlag(bool val)
{
    util::FLAG_REPLICATE_NEW_DEPTH=val;
    util::FLAG_INITIALIZE_NONZERO_POSE=val;

    printf("\nChanging max iterations");
    if(val) //when using initialization
    {
        util::MAX_ITER[0]=5;
        util::MAX_ITER[1]=1;
        util::MAX_ITER[2]=1;
        util::MAX_ITER[3]=1;
    }
    else
    {
        util::MAX_ITER[0]=4;
        util::MAX_ITER[1]=7;
        util::MAX_ITER[2]=9;
        util::MAX_ITER[3]=12;
    
    }
    
}


//for detecting loop closure
inline void loopClosureFlag(bool val)
{
    util::FLAG_DO_LOOP_CLOSURE=val;
    util::FLAG_DO_PARALLEL_SHORT_LOOP_CLOSURE=val; //LC
    util::FLAG_DO_CONST_WEIGHT_POSE_ESTIMATION=val;  //LC
    util::FLAG_DO_PARALLEL_CONST_WEIGHT_POSE_EST=val;  //LC
    
}

//save dpeth mats
inline void saveDepthMatFlag(bool val)
{
    ::util::FLAG_SAVE_MATS=val;
}

void updatePropagationCount()
{
    printf("\nUpdating RA Propagation count");
    
    if(util::ROTATION_AVERAGING_FLAG_ON)
    {
        printf("\nUpdating RA Propagation count");
        util::NUM_RA_PROPAGATION++;
    }
    if(util::GAUSS_NEWTON_FLAG_ON)
    {
        printf("\nUpdating GN Propagation count");
        util::NUM_GN_PROPAGATION++;
    }
}


inline void printFlags()
{
    cout<<"\n*******Printing Flags******\n";
    cout<<"\nFLAG_IS_BOOTSTRAP: "<<util::FLAG_IS_BOOTSTRAP;
    
    cout<<"\nGAUSS_NEWTON_FLAG_ON: "<<util::GAUSS_NEWTON_FLAG_ON;
    cout<<"\nROTATION_AVERAGING_FLAG_ON: "<<util::ROTATION_AVERAGING_FLAG_ON;
    
    cout<<"\nNUM_GN_PROPAGATION: "<<util::NUM_GN_PROPAGATION;
    cout<<"\nNUM_RA_PROPAGATION: "<<util::NUM_RA_PROPAGATION;
    
    cout<<"\nBatch size: "<<util::BATCH_SIZE;
    cout<<"\nBatch start id: "<<util::BATCH_START_ID;
    
    cout<<"\nReplicate New Depth: "<<util::FLAG_REPLICATE_NEW_DEPTH;
    cout<<"\nNon-Zero Initialization: "<<util::FLAG_INITIALIZE_NONZERO_POSE;
    cout<<"\nSave Mat: "<<util::FLAG_SAVE_MATS;
    cout<<"\nLoop Closure: "<<util::FLAG_DO_LOOP_CLOSURE;
    
}



inline bool checkExitCondition()
{
    printf("\n\n\nChecking exit condition");
    if(util::FLAG_IS_BOOTSTRAP)
    {
        printf("\nDetected bootsrap..");
        if(util::NUM_GN_PROPAGATION==0)
        {
            printf("\nSwitching to GN");
            printf("\nTurning OFF Replicate New Depth... ");
            printf("\nTurning ON Save Depth Mats & Loop Closure... ");

            
            util::GAUSS_NEWTON_FLAG_ON=true;
            util::ROTATION_AVERAGING_FLAG_ON=false;
            //for bootstrap, no replicate, only save
            replicateNewDepthFlag(false); //this allows random init of depth map
            saveDepthMatFlag(true);
            loopClosureFlag(true);
            
        }
        else if(util::NUM_GN_PROPAGATION==5*util::KEYFRAME_PROPAGATE_INTERVAL)
        {
            printf("\nTurning OFF Save Depth Mats");
            saveDepthMatFlag(false);
        }
        
        if(util::NUM_GN_PROPAGATION==util::BATCH_SIZE)
        {
            printf("\nMax batch size reached. Terminating..");
            return true; //exit now
        }
        
    }
    else //no bootsrap
    {
        printf("\nNo Bootstrap Detected ..");

        
    //no propagations yet
    if(util::NUM_RA_PROPAGATION==0 && util::NUM_GN_PROPAGATION==0)
    {
        printf("\nSwitching to RA");
        printf("\nTurning ON Replicate New Depth... ");
        printf("\nTurning OFF Save Depth Mats & Loop Closure... ");
        
        util::GAUSS_NEWTON_FLAG_ON=false;
        util::ROTATION_AVERAGING_FLAG_ON=true;
        
        //turn on RA
        replicateNewDepthFlag(true);
        saveDepthMatFlag(false);
        loopClosureFlag(false);
    }
   
    //RA propagations done
    else if(util::NUM_RA_PROPAGATION==util::BATCH_SIZE && util::NUM_GN_PROPAGATION==0)
    {
        printf("\nSwitching to GN");
        printf("\nTurning OFF Replicate New Depth... ");
        printf("\nTurning ON Save Depth Mats & Loop Closure... ");
        
        util::GAUSS_NEWTON_FLAG_ON=true;
        util::ROTATION_AVERAGING_FLAG_ON=false;
        
        //turn on GN
        replicateNewDepthFlag(false);
        saveDepthMatFlag(true);
        loopClosureFlag(true);
    }
    
    //RA propagations already done, GN first propagation done
    else if(util::NUM_RA_PROPAGATION==util::BATCH_SIZE && util::NUM_GN_PROPAGATION>=1)
    {
        printf("\nTurning OFF Save Depth Mats... ");
        
        //turn off save mat
        saveDepthMatFlag(false);
    }
    
        
    //both RA and GN propagations done
    if(util::NUM_RA_PROPAGATION==util::BATCH_SIZE && util::NUM_GN_PROPAGATION==util::BATCH_SIZE)
    {
        printFlags();
        printf("\nMax batch size reached. Terminating..");
        
        return true; //exit now
    }
        
    }
    
    printFlags();
    return false;
    
}

#endif
