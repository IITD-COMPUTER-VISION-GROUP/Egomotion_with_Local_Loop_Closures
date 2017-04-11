/**
 * This file is a part of Egomotion with Local Loop Closures(ELLC) project.
 * For more information visit: http://www.cse.iitd.ac.in/~suvam/wacv2017.html
 **/

#pragma once

#ifndef __DepthHypothesis__
#define __DepthHypothesis__


// This structure contains the variables associated with the depth of a pixel in the current depth map

struct depthhypothesis
{        
    int pixId;
    
    float invDepth=0.0f;//stores inv depth
    
    float invDepthSmoothed=0.0f; //after regularization

    float variance=0.0f;
    
    float varianceSmoothed=0.0f; //after regularization
    
    bool isValid=false;
    
    bool isPerfect=false; //not used
    
    int nUsed=0;
    
    int oldestFrameId=1;
    
    int successful=0;
    
    int validity_counter=0;
    
    int blacklisted=0;
    
};

#endif


