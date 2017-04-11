/**
 * This file is a part of Egomotion with Local Loop Closures(ELLC) project.
 * For more information visit: http://www.cse.iitd.ac.in/~suvam/wacv2017.html
 **/

#ifndef __Tests__
#define __Tests__

#include <stdio.h>
#include "Pyramid.h"
#include "Homography.h"
#include "UserDefinedFunc.h"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

namespace Tests {


    void calculateReprojectionError(frame* prev_frame, frame* current_frame, float* poseCurrentWrtPrev)
    {
        int minHessian=400;
        
        Ptr<SURF> detector = SURF::create(minHessian);
        
        std::vector<KeyPoint> keypoints_prev, keypoints_current;
        detector->detect( prev_frame->image, keypoints_prev );
        detector->detect( current_frame->image, keypoints_current);
        
        Ptr<SURF> extractor = SURF::create();
        Mat descriptors_prev;
        Mat descriptors_current;
        
        extractor->compute( prev_frame->image, keypoints_prev, descriptors_prev );
        extractor->compute( current_frame->image, keypoints_current, descriptors_current);
        
        FlannBasedMatcher matcher;
        vector< DMatch > matches;
        
        if ( descriptors_prev.empty() )
            cvError(0,"MatchFinder","1st descriptor empty",__FILE__,__LINE__);
        if ( descriptors_current.empty() )
            cvError(0,"MatchFinder","2nd descriptor empty",__FILE__,__LINE__);
        
        matcher.match( descriptors_prev, descriptors_current, matches );
        double max_dist = 0; double min_dist = 100;
        
        //-- Quick calculation of max and min distances between keypoints
        int i;
        for(i = 0; i < descriptors_prev.rows; i++ )
        { double dist = matches[i].distance;
            if( dist < min_dist ) min_dist = dist;
            if( dist > max_dist ) max_dist = dist;
        }
        
        printf("-- Max dist : %f \n", max_dist );
        printf("-- Min dist : %f \n", min_dist );
        
        //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
        vector< DMatch > good_matches;
        
        for( i = 0; i < descriptors_prev.rows; i++ )
        { 
            if(matches[i].distance < 3*min_dist)
            { 
                good_matches.push_back( matches[i]); 
            }
        }
        
        Mat img_matches;
        drawMatches( prev_frame->image, keypoints_prev, current_frame->image, keypoints_current,
                    good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                    vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        
        //-- Localize the object
        vector<Point2f> prev; //matched points coordinates
        vector<Point2f> current;
        
        for( i = 0; i < good_matches.size(); i++ )
        {
            //-- Get the keypoints from the good matches
            prev.push_back( keypoints_prev[ good_matches[i].queryIdx ].pt );
            current.push_back( keypoints_current[ good_matches[i].trainIdx ].pt );
        }
        
        
        //---Calculate Reprojection Error going from Prev to Current
        
        //initialize
        vector<float> reprojectionError1;
        reprojectionError1.reserve(good_matches.size());
        
        prev_frame->updationOnPyrChange(0);
        prev_frame->calculateNonZeroDepthPts();
        
        float worldpointX;
        float worldpointY;
        float worldpointZ;
        
        float trfm_worldpointX;
        float trfm_worldpointY;
        float trfm_worldpointZ;
        
        float warpedpointx;
        float warpedpointy;

        //*******CALCULATE RESIZED INTRINSIC PARAMETERS*******//
        
        vector<float> resized_intrinsic= GetIntrinsic(prev_frame->pyrLevel);
        float resized_fx=resized_intrinsic[0];
        float resized_fy=resized_intrinsic[1];
        float resized_cx=resized_intrinsic[2];
        float resized_cy=resized_intrinsic[3];
        

        //SE3
        
        //Create matrix in OpenCV
        Mat se3=(Mat_<float>(4, 4) << 0,-poseCurrentWrtPrev[2],poseCurrentWrtPrev[1],poseCurrentWrtPrev[3], poseCurrentWrtPrev[2],0,-poseCurrentWrtPrev[0],poseCurrentWrtPrev[4], -poseCurrentWrtPrev[1],poseCurrentWrtPrev[0],0,poseCurrentWrtPrev[5],0,0,0,0);
        // Map the OpenCV matrix with Eigen:
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> se3_Eigen(se3.ptr<float>(), se3.rows, se3.cols);
        // Take exp in Eigen and store in new Eigen matrix
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> SE3_Eigen = se3_Eigen.exp();
        // create an OpenCV Mat header for the Eigen data:
        Mat SE3(4, 4, CV_32FC1, SE3_Eigen.data());
        
        float* SE3_ptr;
        float SE3_vec[12];  //r11, r12, r13, t1, r21 r22, r23, t2, r31, r32, r33, t3
        int vec_counter=0;
        for(i=0; i<3; ++i)
        {
            SE3_ptr=SE3.ptr<float>(i); //get pointer to first row of SE3
            SE3_vec[vec_counter++]=SE3_ptr[0];
            SE3_vec[vec_counter++]=SE3_ptr[1];
            SE3_vec[vec_counter++]=SE3_ptr[2];
            SE3_vec[vec_counter++]=SE3_ptr[3];
            
        }
        
        float error1;
        float error1Sum=0.0f;
        
        int k;
        for(k=0;k<good_matches.size(); k++)
        {
            int x=int(prev[k].x);
            int y=int(prev[k].y);
            
            int matchx=int(current[k].x);
            int matchy=int(current[k].y);
            
            
            if(prev_frame->mask.at<uchar>(x,y) ==0)
            {   error1=-1;
                reprojectionError1.push_back(error1);
                continue;
            }
            
            float depth= prev_frame->depth.at<float>(x,y);
            
            //populating world point with homogeneous coordinate ( X ;Y ;depth ; 1)
            worldpointX=(x-resized_cx)*depth/resized_fx; //X-coordinate
            worldpointY=(y-resized_cy)*depth/resized_fy; //Y-coordinate
            worldpointZ=depth; //depth of point at (u,v)
            
            //calculating transformed world point and warped point
            if (SE3_vec[1]==0)
            {
                 trfm_worldpointX=((SE3_vec[0]*worldpointX)+(SE3_vec[1]*worldpointY)+(SE3_vec[2]*worldpointZ)+(SE3_vec[3]));
                 trfm_worldpointY=((SE3_vec[4]*worldpointX)+(SE3_vec[5]*worldpointY)+(SE3_vec[6]*worldpointZ)+(SE3_vec[7]));
                 trfm_worldpointZ=((SE3_vec[8]*worldpointX)+(SE3_vec[9]*worldpointY)+(SE3_vec[10]*worldpointZ)+(SE3_vec[11]));
                
                 warpedpointx=((trfm_worldpointX/trfm_worldpointZ)*resized_fx)+resized_cx;
                 warpedpointy=((trfm_worldpointY/trfm_worldpointZ)*resized_fy)+resized_cy;
            }
            else
            {
                 trfm_worldpointX=(float(SE3_vec[0]*worldpointX)+float(SE3_vec[1]*worldpointY)+float(SE3_vec[2]*worldpointZ)+float(SE3_vec[3]));
                 trfm_worldpointY=(float((SE3_vec[4]*worldpointX))+float((SE3_vec[5]*worldpointY))+float((SE3_vec[6]*worldpointZ))+float((SE3_vec[7])));
                 trfm_worldpointZ=(float(SE3_vec[8]*worldpointX)+float(SE3_vec[9]*worldpointY)+float(SE3_vec[10]*worldpointZ)+float(SE3_vec[11]));
                
                 warpedpointx=float(((trfm_worldpointX/trfm_worldpointZ)*resized_fx)+resized_cx);
                 warpedpointy=float(((trfm_worldpointY/trfm_worldpointZ)*resized_fy)+resized_cy);
                
            }

            error1=pow(matchx-warpedpointx,2);
            error1+=pow(matchy-warpedpointy,2);
            
            error1Sum+=error1;
            
            //printf("\n(x,y)= (%d,%d), (warpedx,warpedy) = (%f, %f), (matchx, matchy)= (%d,%d), depth= %f, error1= %f", x,y, warpedpointx, warpedpointy, matchx, matchy, depth, error1);
            reprojectionError1.push_back(error1);
            
            
        }
        
        cout<<"\nReprojection Error: ";
        for(i=0;i<good_matches.size();i++)
            cout<<reprojectionError1[i]<<"  , ";
        
        cout<<"\n\nReprojection Error Sum: "<<error1Sum;
        
        
    }
    
}

#endif /* defined(__Tests__) */
