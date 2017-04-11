/**
 * This file is a part of Egomotion with Local Loop Closures(ELLC) project.
 * For more information visit: http://www.cse.iitd.ac.in/~suvam/wacv2017.html
 **/
#include "GlobalOptimize.h"

using namespace cv;
using namespace std;

globalOptimize::globalOptimize(string matchfilepath)
{
    match_file.open(matchfilepath);
    
    currentArrayId=0;
    nextArrayId=1;
    isloopClosureDetected=false;
    loopClosureTerminated=false;
    loopClosureArrayId=-1;
    
    firstTestedLoopClosureArrayId=-1;
    lastTestedLoopClosureArrayId=-1;
    
    hsize=256;
    hrange[0]=0;
    hrange[1]=256;
    hist_h=400;
    hist_w=512;
    bin_w=cvRound( (double) hist_w/hsize );
    
    match_window_beg=0;
    match_window_end=util::MAX_LOOP_ARRAY_LENGTH-1;
    
    detectedShortLoopClosure=false;
    
    connectionLost=false;
    temp_depthMap=NULL;
    
}

void globalOptimize::calculateImageHistogram(frame *currentframe)
{
    //printf("\nCalculating image histogram with Id: %d", currentframe->frameId);
    
    const float *hranges[]={hrange};
    isloopClosureDetected=false;
    loopClosureArrayId=-1;
    
    //initializing current loop frame
    currentLoopFrame.isValid=true;
    currentLoopFrame.frameId=currentframe->frameId;
    currentLoopFrame.poseWrtWorld[0]=currentframe->poseWrtWorld[0];
    currentLoopFrame.poseWrtWorld[1]=currentframe->poseWrtWorld[1];
    currentLoopFrame.poseWrtWorld[2]=currentframe->poseWrtWorld[2];
    currentLoopFrame.poseWrtWorld[3]=currentframe->poseWrtWorld[3];
    currentLoopFrame.poseWrtWorld[4]=currentframe->poseWrtWorld[4];
    currentLoopFrame.poseWrtWorld[5]=currentframe->poseWrtWorld[5];
    
    currentLoopFrame.poseWrtOrigin[0]=currentframe->poseWrtOrigin[0];
    currentLoopFrame.poseWrtOrigin[1]=currentframe->poseWrtOrigin[1];
    currentLoopFrame.poseWrtOrigin[2]=currentframe->poseWrtOrigin[2];
    currentLoopFrame.poseWrtOrigin[3]=currentframe->poseWrtOrigin[3];
    currentLoopFrame.poseWrtOrigin[4]=currentframe->poseWrtOrigin[4];
    currentLoopFrame.poseWrtOrigin[5]=currentframe->poseWrtOrigin[5];
    
    currentLoopFrame.image=currentframe->image.clone();
    
    //calculating histogram
    calcHist(&currentframe->image, 1, 0, Mat(), currentLoopFrame.image_histogram, 1, &hsize, hranges, true, false);
    
    //printf("\nHistogram: \n");
    //cout<<currentLoopFrame.image_histogram;
    
    // showImageHistogram(currentLoopFrame.image_histogram, "current normalized histogram");
    
 //   normalize(currentLoopFrame.image_histogram, currentLoopFrame.image_histogram, 0, hist_h, NORM_MINMAX, -1, Mat());
   
    
   
    float sum=0;
    int i;
    for(i=0;i<hsize;i++)
    {
        sum+=currentLoopFrame.image_histogram.at<float>(i);
    }
    
    for(i=0;i<hsize;i++)
    {
       currentLoopFrame.image_histogram.at<float>(i)/=sum;
    }
    
    
    //printf("\nNormalized histogram: \n");
    //cout<<currentLoopFrame.image_histogram;
    
    //MatND normalizeHist;
    //normalize(image_histogram[start_idx], normalizeHist, 0, hist_h, NORM_MINMAX, -1, Mat());
    
   
    
}

void globalOptimize::showImageHistogram(MatND histogram_normalized, string name)
{
    Mat histImage( hist_h, hist_w, CV_8UC1, Scalar( 0,0,0) );
    int i;
    for( i = 1; i < hsize; i++ )
    {
        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(histogram_normalized.at<float>(i-1)) ) ,
             Point( bin_w*(i), hist_h - cvRound(histogram_normalized.at<float>(i)) ),
             Scalar( 255, 0, 0), 2, 8, 0  );
    }
    
    imshow( name, histImage );
}

double globalOptimize::compareImageHistogram(MatND first_histogram_norm, MatND second_histogram_norm)
{
   
    return compareHist(first_histogram_norm, second_histogram_norm, CV_COMP_KL_DIV);
 
    
}

void globalOptimize::resetArrayElement(int arrayId)
{
    
    //printf("\nResetting arrayId: %d with frameI Id: %d", arrayId, loopFrameArray[arrayId].this_frame->frameId);
    
    //clear loop frame element
    delete loopFrameArray[arrayId].this_frame;
    loopFrameArray[arrayId].this_frame=NULL;
    
    if(!loopFrameArray[arrayId].isStray)
    {   
        delete loopFrameArray[arrayId].this_currentDepthMap;
        loopFrameArray[arrayId].this_currentDepthMap=NULL;
    }
    
    loopFrameArray[arrayId].isValid=false;
    loopFrameArray[arrayId].isStray=false;
    loopFrameArray[arrayId].frameId=0;
    

    
    //delete loopFrameArray[arrayId].new_frame;
    //loopFrameArray[arrayId].new_frame=NULL;
    
}


void globalOptimize::pushToArray(frame *currentframe, depthMap* currentDepthMap)
{
    //boost::thread_group t_group; //this thread will run parallel to main program thread.
    
    if(util::FLAG_DO_PARALLEL_SHORT_LOOP_CLOSURE)
    {
        util::measureTime wait_clock;
        wait_clock.startTimeMeasure();
        //printf("\nIn push to Array. Waiting for match thread...");
        
        t_group.join_all(); //wait here for match thread to complete before pushing another frame in loop array
        
        float wait_time=wait_clock.stopTimeMeasure();
        printf(" for %f ms", wait_time);
        
        if(wait_time>0.0f)
        {
            printf("\nYESS");
        }
    }
    
    
 
    
    //Step 1 -> push test-frame in array since possibility of it getting removed from frameptr_vector, but dont update pther variables
    

    printf("\n\nPushing frame with Id: %d at loop array id: %d \n",currentframe->frameId, currentArrayId);
    
   if(loopFrameArray[currentArrayId].isValid==true)
            resetArrayElement(currentArrayId);

    calculateImageHistogram(currentframe);
    
    loopFrameArray[currentArrayId].this_frame=new frame(*currentframe);
    loopFrameArray[currentArrayId].this_currentDepthMap=new depthMap(*currentDepthMap);
    

    
    loopFrameArray[currentArrayId].frameId=currentLoopFrame.frameId;  //currentframe->frameId;
    loopFrameArray[currentArrayId].isValid=currentLoopFrame.isValid;//true;
    
    loopFrameArray[currentArrayId].image=currentLoopFrame.image.clone();//currentframe->image.clone();
    
    loopFrameArray[currentArrayId].image_histogram=currentLoopFrame.image_histogram.clone();
    
    loopFrameArray[currentArrayId].poseWrtWorld[0]=currentframe->poseWrtWorld[0];
    loopFrameArray[currentArrayId].poseWrtWorld[1]=currentframe->poseWrtWorld[1];
    loopFrameArray[currentArrayId].poseWrtWorld[2]=currentframe->poseWrtWorld[2];
    loopFrameArray[currentArrayId].poseWrtWorld[3]=currentframe->poseWrtWorld[3];
    loopFrameArray[currentArrayId].poseWrtWorld[4]=currentframe->poseWrtWorld[4];
    loopFrameArray[currentArrayId].poseWrtWorld[5]=currentframe->poseWrtWorld[5];
    
    loopFrameArray[currentArrayId].poseWrtOrigin[0]=currentframe->poseWrtOrigin[0];
    loopFrameArray[currentArrayId].poseWrtOrigin[1]=currentframe->poseWrtOrigin[1];
    loopFrameArray[currentArrayId].poseWrtOrigin[2]=currentframe->poseWrtOrigin[2];
    loopFrameArray[currentArrayId].poseWrtOrigin[3]=currentframe->poseWrtOrigin[3];
    loopFrameArray[currentArrayId].poseWrtOrigin[4]=currentframe->poseWrtOrigin[4];
    loopFrameArray[currentArrayId].poseWrtOrigin[5]=currentframe->poseWrtOrigin[5];
    
    //updating keyframe to point to loopframe
    loopFrameArray[currentArrayId].this_currentDepthMap->keyFrame=loopFrameArray[currentArrayId].this_frame;
    
    loopFrameArray[currentArrayId].this_currentDepthMap->depthvararrptr[0]=loopFrameArray[currentArrayId].this_currentDepthMap->depthvararrpyr0;
    loopFrameArray[currentArrayId].this_currentDepthMap->depthvararrptr[1]=loopFrameArray[currentArrayId].this_currentDepthMap->depthvararrpyr1;
    loopFrameArray[currentArrayId].this_currentDepthMap->depthvararrptr[2]=loopFrameArray[currentArrayId].this_currentDepthMap->depthvararrpyr2;
    loopFrameArray[currentArrayId].this_currentDepthMap->depthvararrptr[3]=loopFrameArray[currentArrayId].this_currentDepthMap->depthvararrpyr3;
    
    
    loopFrameArray[currentArrayId].this_currentDepthMap->deptharrptr[0]=loopFrameArray[currentArrayId].this_currentDepthMap->deptharrpyr0;
    loopFrameArray[currentArrayId].this_currentDepthMap->deptharrptr[1]=loopFrameArray[currentArrayId].this_currentDepthMap->deptharrpyr1;
    loopFrameArray[currentArrayId].this_currentDepthMap->deptharrptr[2]=loopFrameArray[currentArrayId].this_currentDepthMap->deptharrpyr2;
    loopFrameArray[currentArrayId].this_currentDepthMap->deptharrptr[3]=loopFrameArray[currentArrayId].this_currentDepthMap->deptharrpyr3;
    
   
        
    if(util::FLAG_USE_LOOP_CLOSURE_TRIGGER)
    {
        triggerRotation(currentframe);
    }
    else
        detectedShortLoopClosure=true;
    
    
    //Step 2 -> find matches and do pose estimation
    
    if(detectedShortLoopClosure)
    {
        if(util::FLAG_DO_PARALLEL_SHORT_LOOP_CLOSURE)
        {
            t_group.create_thread(boost::bind(&globalOptimize::findMatchParallel,this, loopFrameArray[currentArrayId].this_frame,1));
            
            if(util::FLAG_ALTERNATE_GN_RA && util::EXIT_CONDITION)
            {
                printf("\nExit Condition already detected! Waiting for loop closure threads to complete....");
                t_group.join_all();
                loopClosureTerminated=true;
                return;
            }
        }

        else
        {
            util::measureTime global_match_time;
            global_match_time.startTimeMeasure();
            findMatchParallel(loopFrameArray[currentArrayId].this_frame,0);
            float time_elapsed=global_match_time.stopTimeMeasure();
            printf("\nTime elapsed in Global Loop closure: %f", time_elapsed);
            
            if(util::FLAG_ALTERNATE_GN_RA && util::EXIT_CONDITION)
            {
                printf("\nExit Condition already detected! NO threads....");
                loopClosureTerminated=true;
                return;
            }
        }
    }
    
    //printLoopArrayStats();
    //printf("\nReturning to main loop from push to loop array");
    return;
}

bool globalOptimize::findMatch(frame *currentframe, bool strayFlag)
{
    //printf("\n\nFinding Match: ");
    calculateImageHistogram(currentframe);
    //lastTestedLoopClosureArrayId=-1;
    int i;
    
    
    
    if(lastTestedLoopClosureArrayId==-1)
    {
        //printf("\n\nMatch window beg: %d, end: %d for test array id: %d", match_window_beg, match_window_end, currentArrayId);
        i=currentArrayId-1; //comparison starts from most recent

        
    }
    else
    {   
        if(lastTestedLoopClosureArrayId!=0)
            i=lastTestedLoopClosureArrayId-1;
        else
            i=util::MAX_LOOP_ARRAY_LENGTH_SCALE_AVG-1;
    }
    
    if(i<0)
    {
        i=util::MAX_LOOP_ARRAY_LENGTH_SCALE_AVG-1;

    }
    
    
    int conditionToTerminateLoop=0;
    
    
    while(1)
    {
        printf("\nComparing (array id %d & current frame Id: %d) with (match array Id: %d & frame id: %d ) ", currentArrayId, currentLoopFrame.frameId, i, loopFrameArray[i].frameId);
        lastTestedLoopClosureArrayId=i;
        
        //check conditions to come out of loop
        
        if (match_window_end > match_window_beg)
        {
            if(!((i>=match_window_beg) && (i<=match_window_end)))
                conditionToTerminateLoop=1;
        }
        else if(match_window_end < match_window_beg)
        {
            if (!((i>=match_window_beg) || (i<=match_window_end)))
                conditionToTerminateLoop=1;
        }
        else if(loopFrameArray[i].isValid==false)
            conditionToTerminateLoop=1;
        
        if(conditionToTerminateLoop==1)
        {
            //printf("\nFrame id: %d is out of Bounds [%d, %d]  or Invalid = %d. Exiting matching Loop!", i, match_window_beg, match_window_end,!loopFrameArray[i].isValid);
            lastTestedLoopClosureArrayId=-1; //no more testing to be done
            break;
        }
        
        
        if(loopFrameArray[i].isValid==0)
        {     //printf(" Loop frame array: %d is not valid!!", i);
            lastTestedLoopClosureArrayId=-1; //no more testing to be done
            return false;
        }
             
        //printf("and keyframe Id: %d)",loopFrameArray[i].this_frame->frameId );
        
        if((currentframe->frameId-loopFrameArray[i].frameId > util::MIN_MATCH_DIFFERENCE) )
        {
        
            //calculating stats
            matchValue=compareImageHistogram(loopFrameArray[i].image_histogram, currentLoopFrame.image_histogram);
            
            if(!strayFlag) //since only have pose estimate for non-stray frames
                calculateRotationStats(loopFrameArray[i].poseWrtWorld, currentLoopFrame.poseWrtWorld);
                
             if(!strayFlag)
                {
                    // imshow("Matched frame", loopFrameArray[i].image);
                    // imshow("Current Frame", currentLoopFrame.image);
                    // printf("\n\nMatch value for arrayId %d is: %f", i, matchValue);
                    // printf("\nRelative view angle: %f\n\n", relative_view_angle);
                    // waitKey(0);
                }
            
            
            
            
            if(matchValue<=util::MATCH_THRESHOLD || strayFlag)
            {
                if(!strayFlag)
                {   
                    if(relative_view_angle<=util::MAX_REL_VIEW_ANGLE) //check angle only for non-stray
                    {   
                        isloopClosureDetected=true;
                        /*
                         imshow("Matched frame", loopFrameArray[i].image);
                         imshow("Current Frame", currentLoopFrame.image);
                         printf("\n\nMatch value for arrayId %d is: %f", i, matchValue);
                         printf("\nRMS error: %f", rms_error);
                         */
                        loopClosureArrayId=i;
                        break; //match found
                    }
                }
                else
                {
                    isloopClosureDetected=true;
                    
                    loopClosureArrayId=i;
                    break; //match found
                    
                }
            }
        }
        else
        {
            //printf("\nSkipping..");
        }
        
        //update i
        i--;
        if(i<0)
            i=util::MAX_LOOP_ARRAY_LENGTH_SCALE_AVG-1;
        
    }

    if(isloopClosureDetected==true)
    {
        //printf("\nMatch found: YES between current frame Id: %d and Loop frame Id: %d ", currentLoopFrame.frameId, loopFrameArray[i].frameId);
    }

    else
    {
        //printf("\nMatch found: NO, last tested loop array: %d", lastTestedLoopClosureArrayId);
    }

    return isloopClosureDetected;
    
}


void globalOptimize::calculateRotationStats(float* pose1_0, float* pose2_0)
{
    rms_error=pow(pow(pose1_0[0]-pose2_0[0],2)+pow(pose1_0[1]-pose2_0[1],2)+pow(pose1_0[2]-pose2_0[2],2),0.5);
    float view_vec1[3];
    float view_vec2[3];
    
    calculateViewVec(pose1_0, view_vec1);
    calculateViewVec(pose2_0, view_vec2);
    float mag1=pow(view_vec1[0]*view_vec1[0]+view_vec1[1]*view_vec1[1]+view_vec1[2]*view_vec1[2],0.5);
    float mag2=pow(view_vec2[0]*view_vec2[0]+view_vec2[1]*view_vec2[1]+view_vec2[2]*view_vec2[2],0.5);
    
    relative_view_angle=acos((view_vec1[0]*view_vec2[0]+view_vec1[1]*view_vec2[1]+view_vec1[2]*view_vec2[2])/(mag1*mag2));
    //convert angle to degrees
    relative_view_angle=(relative_view_angle*180)/3.14f;

}

void globalOptimize::calculateViewVec(float* pose, float* view_vec)
{
    //Create matrix in OpenCV
    Mat se3=(Mat_<float>(4, 4) << 0,-pose[2],pose[1],pose[3], pose[2],0,-pose[0],pose[4], -pose[1],pose[0],0,pose[5],0,0,0,0);
    // Map the OpenCV matrix with Eigen:
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> se3_Eigen(se3.ptr<float>(), se3.rows, se3.cols);
    // Take exp in Eigen and store in new Eigen matrix
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> SE3_Eigen = se3_Eigen.exp(); //4x4 pose of Other wrt This (eigen)
    // rotation matrix
    Eigen::MatrixXf SE3_R=SE3_Eigen.block(0, 0, 3, 3);
    
    view_vec[0]=SE3_R(2,0);
    view_vec[1]=SE3_R(2,1);
    view_vec[2]=SE3_R(2,2);
    return;
    
}

void globalOptimize::findMatchParallel(frame *testFrame, int thread_num)
{
    printf("\nThread no: %d, In find match parallel loop closure: %d for frame id: %d at array id: %d", thread_num, util::FLAG_DO_PARALLEL_SHORT_LOOP_CLOSURE, testFrame->frameId, currentArrayId);
    
    /*printf("\nWaiting...");
    waitKey(10000);
    printf("\nWait over!!");*/
    
    //check for loop closures
        bool matchFound=false;
        int waitFrameCount=0;
        vector<float>pose_vec;
        int seeds_num;
    
    
        if(waitFrameCount!=0)
        {    waitFrameCount--;
            //printf("\nNot checking loop closure");
        }
        
        if( /*(frame_counter%util::KEYFRAME_PROPAGATE_INTERVAL==0) && */ waitFrameCount==0)
        {
            loopFrameArray[nextArrayId].frameId=testFrame->frameId;
            lastTestedLoopClosureArrayId=-1;
            firstTestedLoopClosureArrayId=-1;
            int num_matches=0;
            do
            {
                //printf("\nchecking for Loop Closures!");
                
                matchFound=findMatch(testFrame);
                //cout<<"\nMatch found: NO ";
                
                
                //  imshow("Current image", frameptr_vector.back()->image);
                //  imshow("this image",globalOptimizeLoop.loopFrameArray[globalOptimizeLoop.loopClosureArrayId].this_frame->image);
                //  imshow("in array image", globalOptimizeLoop.loopFrameArray[globalOptimizeLoop.loopClosureArrayId].image);
                //  waitKey(0);
                
                if(num_matches>0 && lastTestedLoopClosureArrayId==firstTestedLoopClosureArrayId)
                {
                    //printf("\nComing out of loop: FirstTestedId: %d, LastTestedId: %d", firstTestedLoopClosureArrayId, lastTestedLoopClosureArrayId);
                    break;
                }
                
                if(matchFound==true)
                {
                    
                    /*
                    if(util::FLAG_SAVE_MATCH_IMAGES)
                    {
                        //to save images
                        
                        stringstream ss;
                        string str = "img_";
                        string str2="_";
                        string type = ".jpg";
                        
                        if(num_matches==0)
                        {
                            //saving current image, name: img_currframeid_0.jpg
                            ss<<matchsave_path<<str<<frameptr_vector.back()->frameId<<str2<<num_matches<<type;
                            string filename = ss.str();
                            ss.str("");
                            imwrite(filename, frameptr_vector.back()->image);
                            
                        }
                        
                        //saving match image, name: img_currframeid_0_matchframeid.jpg
                        ss<<matchsave_path<<str<<frameptr_vector.back()->frameId<<str2<<globalOptimizeLoop.loopFrameArray[globalOptimizeLoop.loopClosureArrayId].this_frame->frameId<<type;
                        string filename = ss.str();
                        ss.str("");
                        imwrite(filename, globalOptimizeLoop.loopFrameArray[globalOptimizeLoop.loopClosureArrayId].this_frame->image);
                    }
                    */
                    
                    //update variables
                    if(num_matches==0)
                        firstTestedLoopClosureArrayId=lastTestedLoopClosureArrayId;
                    
                    num_matches++;
                    matchFound=false;
                    waitFrameCount=util::MIN_WAIT_COUNT;
                    
                    //cout<<"\n"<<frameptr_vector.back()->frameId<<" "<<globalOptimizeLoop.loopFrameArray[globalOptimizeLoop.loopClosureArrayId].this_frame->frameId;
                    
                    
                    /*
                     str="currentimg_";
                     //saving current image
                     ss<<matchsave_path<<str<<(frameptr_vector.back()->frameId)<<type;
                     filename = ss.str();
                     ss.str("");
                     imwrite(filename, frameptr_vector.back()->image);
                     */
                    
                    
                    //calculate pose with matched frame
                    
                    float initial_pose[6];
                    if(util::FLAG_INITIALIZE_NONZERO_POSE)
                    {
                        initial_pose[0]=0.0f;
                        initial_pose[1]=0.0f;
                        initial_pose[2]=0.0f;
                        initial_pose[3]=0.0f;
                        initial_pose[4]=0.0f;
                        initial_pose[5]=0.0f;
                        
                    }
                    
                    
                    pose_vec=GetImagePoseEstimate(loopFrameArray[loopClosureArrayId].this_frame,
                                                  testFrame, testFrame->frameId, loopFrameArray[loopClosureArrayId].this_currentDepthMap,
                                                  testFrame, initial_pose, true);
                    
                    seeds_num= loopFrameArray[loopClosureArrayId].this_currentDepthMap->calculate_no_of_Seeds();
                    
                    
                    
                    if(util::FLAG_WRITE_MATCH_POSES)
                    {
                        
                        //pose wrt origin poses-> kf id, frame id
                        if(match_file.is_open())
                        {
                            match_file<<(testFrame->frameId+util::BATCH_START_ID-1)<<" "<<(loopFrameArray[loopClosureArrayId].this_frame->frameId+util::BATCH_START_ID-1)<<" "<<testFrame->poseWrtOrigin[0]<<" "<<testFrame->poseWrtOrigin[1]<<" "<<testFrame->poseWrtOrigin[2]<<" "<<testFrame->poseWrtOrigin[3]<<" "<<testFrame->poseWrtOrigin[4]<<" "<<testFrame->poseWrtOrigin[5]<<" "<<loopFrameArray[loopClosureArrayId].this_frame->rescaleFactor<<" "<<seeds_num<<" "<<matchValue<<" "<<rms_error<<" "<<relative_view_angle<<"\n";
                        }
                    }
                    
                    
                    
                    
                    cout<<"\n"<<testFrame->frameId<<" "<<loopFrameArray[loopClosureArrayId].this_frame->frameId<<" "<<testFrame->poseWrtWorld[0]<<" "<<testFrame->poseWrtWorld[1]<<" "<<testFrame->poseWrtWorld[2]<<" "<<testFrame->poseWrtWorld[3]<<" "<<testFrame->poseWrtWorld[4]<<" "<<testFrame->poseWrtWorld[5]<<" "<<loopFrameArray[loopClosureArrayId].this_frame->rescaleFactor<<" "<<seeds_num;
                    
                    //updating to original pose
                    
                    testFrame->poseWrtWorld[0]=currentLoopFrame.poseWrtWorld[0];
                    testFrame->poseWrtWorld[1]=currentLoopFrame.poseWrtWorld[1];
                    testFrame->poseWrtWorld[2]=currentLoopFrame.poseWrtWorld[2];
                    testFrame->poseWrtWorld[3]=currentLoopFrame.poseWrtWorld[3];
                    testFrame->poseWrtWorld[4]=currentLoopFrame.poseWrtWorld[4];
                    testFrame->poseWrtWorld[5]=currentLoopFrame.poseWrtWorld[5];
                    
                    
                    //updating to original pose
                    
                    testFrame->poseWrtOrigin[0]=currentLoopFrame.poseWrtOrigin[0];
                    testFrame->poseWrtOrigin[1]=currentLoopFrame.poseWrtOrigin[1];
                    testFrame->poseWrtOrigin[2]=currentLoopFrame.poseWrtOrigin[2];
                    testFrame->poseWrtOrigin[3]=currentLoopFrame.poseWrtOrigin[3];
                    testFrame->poseWrtOrigin[4]=currentLoopFrame.poseWrtOrigin[4];
                    testFrame->poseWrtOrigin[5]=currentLoopFrame.poseWrtOrigin[5];
                    
                }
                
            }while (lastTestedLoopClosureArrayId!=-1);
            
        }
    
    //Step 3 -> Now update variables
   
    //updating current array id-> position where next frame will be pushed
    currentArrayId++;
    nextArrayId++;
    
    
    //updating match window end and beginning ids
    
    if(currentArrayId==match_window_end+2 )
    {    
        match_window_beg++;
        match_window_end++;
    }
    if(currentArrayId==1 && match_window_end==util::MAX_LOOP_ARRAY_LENGTH_SCALE_AVG-1)
    {    
        match_window_beg++;
        match_window_end=0;
    }
    
    if(currentArrayId==util::MAX_LOOP_ARRAY_LENGTH_SCALE_AVG) //reset currentArrayId
        currentArrayId=0;
    if(nextArrayId==util::MAX_LOOP_ARRAY_LENGTH_SCALE_AVG) //reset currentArrayId
        nextArrayId=0;
    if(match_window_end==util::MAX_LOOP_ARRAY_LENGTH_SCALE_AVG)
        match_window_end=0;
    if(match_window_beg==util::MAX_LOOP_ARRAY_LENGTH_SCALE_AVG)
        match_window_beg=0;
    
    //printf("\nNOW: currentArrayId: %d, match_beg: %d. match_end: %d \n", currentArrayId, match_window_beg, match_window_end);

    //printf("\nReturning from match parallel with thread no: %d", thread_num);
}


void globalOptimize::printLoopArrayStats()
{
    printf("\n\nSTATS->Index Variables: \nCurrentArrayId: %d, NextArrayId: %d, \nFirstTestedLoopClosureArrayId: %d,  lastTestedLoopClosureArrayId: %d \nMax loop array size: %d, %d", currentArrayId, nextArrayId, firstTestedLoopClosureArrayId, lastTestedLoopClosureArrayId, util::MAX_LOOP_ARRAY_LENGTH, util::MAX_LOOP_ARRAY_LENGTH_SCALE_AVG);
    
    printf("\nIterating over loop from first to last element...");
    bool found=false;
    int i;
    for(i=0; i<util::MAX_LOOP_ARRAY_LENGTH_SCALE_AVG; i++)
    {
        if(loopFrameArray[i].isValid==false)
            continue;
       
        if(!found)
            found=true;
        printf("\nArray id: %d, Frame id: %d", i, loopFrameArray[i].frameId+util::BATCH_START_ID-1);
    }
    
    if(!found)
        printf("\nNo valid array elements!\n");
}


void globalOptimize::triggerRotation(frame* currentframe)
{
    imshow("Current Frame", currentframe->image);
    waitKey(2000);
    
    currentframe->calculateRelativeRandT(currentframe->poseWrtWorld);
    Eigen::Vector3f center=-currentframe->SE3_R.transpose()*currentframe->SE3_T;
    Eigen::Vector3f viewvector=Eigen::Vector3f(currentframe->SE3_R(2,0),currentframe->SE3_R(2,1), currentframe->SE3_R(2,2));
    
    //theta=acos(a.b/|a||b|)
    float theta=center.dot(viewvector);
    theta=theta/(center.norm()*viewvector.norm());
    theta=(acos(theta)*180)/3.14f;
    
    //cross product
    Eigen::Vector3f cross_vec=center.cross(viewvector);
    
    cout<<"\nIn Trigger: "<<"theta: "<<theta;//<<", cross\n"<<cross_vec<<"\n\n";
    
    if(!detectedShortLoopClosure) //if no loop closure, then check condition to turn on
    {
        if(theta>util::TRIGGER_LOOP_CLOSURE_ON)
           detectedShortLoopClosure=true;
        
    }
    
    if(detectedShortLoopClosure) //if loop closure on, then check condition to turn it off
    {
        if(theta<util::TRIGGER_LOOP_CLOSURE_OFF)
        {
            detectedShortLoopClosure=false;
            
            if(util::FLAG_WRITE_MATCH_POSES)
            {
                //indicate loop cloure is now switching off
                if(match_file.is_open())
                {
                    match_file<<"-1"<<"\n";
                }
            }
        }
    }

    
}

void globalOptimize::findConnection(frame *testFrame)
{
    printf("\nIn find connection for frame id: %d at array id: %d", testFrame->frameId, currentArrayId);
    
   
    //Step 1->wait for any previous thread to complete, but dont return to main yet!
    if(util::FLAG_DO_PARALLEL_SHORT_LOOP_CLOSURE)
    {
        util::measureTime wait_clock;
        wait_clock.startTimeMeasure();
        printf("\nWaiting for match thread...");
        
        t_group.join_all(); //wait here for match thread to complete before pushing another frame in loop array
        
        float wait_time=wait_clock.stopTimeMeasure();
        printf(" for %f ms", wait_time);
        
        if(wait_time>0.0f)
        {
            printf("\nYESS");
        }
    }
    
    //Step 2->push stray test frame(no depth map) to the array at currentArrayId
    
    //printLoopArrayStats();
    printf("\n\nPushing frame with Id: %d at loop array id: %d \n",testFrame->frameId, currentArrayId);
    
    if(loopFrameArray[currentArrayId].isValid==true)
        resetArrayElement(currentArrayId);
    
    calculateImageHistogram(testFrame);
    
    loopFrameArray[currentArrayId].isStray=true; //since has no depth map!
    loopFrameArray[currentArrayId].this_frame=new frame(*testFrame);
    
    loopFrameArray[currentArrayId].frameId=currentLoopFrame.frameId;  //currentframe->frameId;
    loopFrameArray[currentArrayId].isValid=currentLoopFrame.isValid;//true;
    
    loopFrameArray[currentArrayId].image=currentLoopFrame.image.clone();//currentframe->image.clone();
    loopFrameArray[currentArrayId].image_histogram=currentLoopFrame.image_histogram.clone();
    
    
    //Step 3 -> find match with a frame (only test for KL div condition)
    //this step finds a matching frame, re-estimates pose of test frame with match frame, propagates depeth map of matched frame to test frame and re-checks condition on seeds. Exits if connection is no longer lost
    //if still connection lost, then finds a new match.
    
    //check for loop closure
    bool matchFound=false;
    int waitFrameCount=0;
    vector<float>pose_vec;
    
    if(waitFrameCount!=0)
    {    waitFrameCount--;
        //printf("\nNot checking loop closure");
    }
    
    if( /*(frame_counter%util::KEYFRAME_PROPAGATE_INTERVAL==0) && */ waitFrameCount==0)
    {
        loopFrameArray[nextArrayId].frameId=testFrame->frameId;
        lastTestedLoopClosureArrayId=-1;
        firstTestedLoopClosureArrayId=-1;
        int num_matches=0;
        do
        {
            printf("\nchecking for Connections!");
            
            matchFound=findMatch(loopFrameArray[currentArrayId].this_frame, loopFrameArray[currentArrayId].isStray);
            
            if(num_matches>0 && lastTestedLoopClosureArrayId==firstTestedLoopClosureArrayId)
            {
                //printf("\nComing out of loop: FirstTestedId: %d, LastTestedId: %d", firstTestedLoopClosureArrayId, lastTestedLoopClosureArrayId);
                break;
            }
            
            if(matchFound==true)
            {
                /*
                 if(util::FLAG_SAVE_MATCH_IMAGES)
                 {
                 //to save images
                 
                 stringstream ss;
                 string str = "img_";
                 string str2="_";
                 string type = ".jpg";
                 
                 if(num_matches==0)
                 {
                 //saving current image, name: img_currframeid_0.jpg
                 ss<<matchsave_path<<str<<frameptr_vector.back()->frameId<<str2<<num_matches<<type;
                 string filename = ss.str();
                 ss.str("");
                 imwrite(filename, frameptr_vector.back()->image);
                 
                 }
                 
                 //saving match image, name: img_currframeid_0_matchframeid.jpg
                 ss<<matchsave_path<<str<<frameptr_vector.back()->frameId<<str2<<globalOptimizeLoop.loopFrameArray[globalOptimizeLoop.loopClosureArrayId].this_frame->frameId<<type;
                 string filename = ss.str();
                 ss.str("");
                 imwrite(filename, globalOptimizeLoop.loopFrameArray[globalOptimizeLoop.loopClosureArrayId].this_frame->image);
                 }
                 */
                
                //update variables
                if(num_matches==0)
                    firstTestedLoopClosureArrayId=lastTestedLoopClosureArrayId;
                
                num_matches++;
                //connectionLost=false; //reset connection lost if match found!
                matchFound=false;
                waitFrameCount=util::MIN_WAIT_COUNT;
                
                //cout<<"\n"<<frameptr_vector.back()->frameId<<" "<<globalOptimizeLoop.loopFrameArray[globalOptimizeLoop.loopClosureArrayId].this_frame->frameId;
                
                
                //cout<<"\n"<<frameptr_vector.back()->frameId<<" "<<globalOptimizeLoop.loopFrameArray[globalOptimizeLoop.loopClosureArrayId].this_frame->frameId;
                
                
                /*
                 str="currentimg_";
                 //saving current image
                 ss<<matchsave_path<<str<<(frameptr_vector.back()->frameId)<<type;
                 filename = ss.str();
                 ss.str("");
                 imwrite(filename, frameptr_vector.back()->image);
                 */
                
                
                //calculate pose with matched frame
                printf("\nFinding pose with matched frame");
                float initial_pose[6];
                if(util::FLAG_INITIALIZE_NONZERO_POSE)
                {
                    initial_pose[0]=0.0f;
                    initial_pose[1]=0.0f;
                    initial_pose[2]=0.0f;
                    initial_pose[3]=0.0f;
                    initial_pose[4]=0.0f;
                    initial_pose[5]=0.0f;
                    
                }
                
                //find pose of test frame with the matched frame
                pose_vec=GetImagePoseEstimate(loopFrameArray[loopClosureArrayId].this_frame,
                                              testFrame, testFrame->frameId, loopFrameArray[loopClosureArrayId].this_currentDepthMap,
                                              testFrame, initial_pose, true);
                
                
                cout<<"\nFIND CONNECTION: "<<testFrame->frameId<<" "<<loopFrameArray[loopClosureArrayId].this_frame->frameId<<" "<<testFrame->poseWrtWorld[0]<<" "<<testFrame->poseWrtWorld[1]<<" "<<testFrame->poseWrtWorld[2]<<" "<<testFrame->poseWrtWorld[3]<<" "<<testFrame->poseWrtWorld[4]<<" "<<testFrame->poseWrtWorld[5]<<" "<<loopFrameArray[loopClosureArrayId].this_frame->rescaleFactor<<" "<<loopFrameArray[loopClosureArrayId].this_currentDepthMap->calculate_no_of_Seeds();
                
                
                //creating temp copy of depth map
                temp_depthMap=new depthMap(*loopFrameArray[loopClosureArrayId].this_currentDepthMap);
                temp_depthMap->keyFrame=loopFrameArray[loopClosureArrayId].this_currentDepthMap->keyFrame;
                temp_depthMap->depthvararrptr[0]=temp_depthMap->depthvararrpyr0;
                temp_depthMap->depthvararrptr[1]=temp_depthMap->depthvararrpyr1;
                temp_depthMap->depthvararrptr[2]=temp_depthMap->depthvararrpyr2;
                temp_depthMap->depthvararrptr[3]=temp_depthMap->depthvararrpyr3;
                
                temp_depthMap->deptharrptr[0]=temp_depthMap->deptharrpyr0;
                temp_depthMap->deptharrptr[1]=temp_depthMap->deptharrpyr1;
                temp_depthMap->deptharrptr[2]=temp_depthMap->deptharrpyr2;
                temp_depthMap->deptharrptr[3]=temp_depthMap->deptharrpyr3;

                printf("\n\n\nkeyframe id: %d",temp_depthMap->keyFrame->frameId);
                //check depth propagation from matched frame to test frame
                printf("\nNo of seeds before propgation: %f",temp_depthMap->calculate_no_of_Seeds());
               
                temp_depthMap->displayColourDepthMap(temp_depthMap->keyFrame);

                temp_depthMap->currentFrame=testFrame;
                temp_depthMap->createKeyFrame(testFrame);
                
                 printf("\nNo of seeds after propgation: %f",temp_depthMap->calculate_no_of_Seeds());
                
                temp_depthMap->displayColourDepthMap(temp_depthMap->keyFrame);
                
                checkConnection(temp_depthMap);
                
                if(connectionLost==true)
                {
                    delete temp_depthMap;
                    temp_depthMap=NULL;
                    continue;
                }
                
                
                
                
                printf("\nAborting matching loop");
                break; //come out of loop if any one match is found
                
            }
            
        }while (lastTestedLoopClosureArrayId!=-1);
        
    }
    
    
    //Step 3 -> Now DON't update variables. Instead just reset!
    resetArrayElement(currentArrayId);
    
   // printLoopArrayStats();
    
    
    
    
    
    //printf("\nNOW: currentArrayId: %d, match_beg: %d. match_end: %d \n", currentArrayId, match_window_beg, match_window_end);
    
    printf("\nReturning from find Connection function");
    return;

}

void globalOptimize::checkConnection(depthMap *currentDepthMap)
{
    //printf("\nChecking if Connection Lost...");
    if(currentDepthMap->calculate_no_of_Seeds()<=util::MIN_SEEDS_FOR_CONNECTION_LOST)
        connectionLost=true;
    else
        connectionLost=false;
    
    //printf(" %d", connectionLost);
}








