/**
 * This file is a part of Egomotion with Local Loop Closures(ELLC) project.
 * For more information visit: http://www.cse.iitd.ac.in/~suvam/wacv2017.html
 **/

#pragma once

#ifndef __DisplayFunc__
#define __DisplayFunc__

#include <cstdio>
#include "opencv2/opencv.hpp"
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include "Frame.h"
#include <string>

using namespace std;
using namespace cv;

#ifdef DEBUG
#define PRINTF(...) //{  printf(__VA_ARGS__); }
#else
#define PRINTF(...)
#endif

#define CHECK_FOR_ERRORS
#ifdef _DEBUG
#define CHECK_FOR_ERRORS
#endif


#ifdef CHECK_FOR_ERRORS
#define DEBUG_COND(cond)	DEBUG_COND_FUNC((cond))
#define DEBUG_STATEMENT(X)	X
#define MY_ASSERT(X)	MY_ASSERT_FUNC(X)

#else
#define DEBUG_STATEMENT(X)
#define DEBUG_COND(cond)
#define MY_ASSERT(X)
#endif

#pragma warning (disable : 4996)

extern FILE* STATE_FILE;
extern void MY_ASSERT_FUNC(bool cond);
extern void DEBUG_COND_FUNC(bool cond);

//for pyramids
void DisplayIterationRes(frame* image_frame, Mat residual, String name, int framenum ,bool homo=false ); //for iteration residual
void DisplayInitialRes(frame* current_frame,frame* prev_frame,String name, int framenum,bool homo=false ); //for initial residual
void DisplayWarpedImg(Mat warpedimg, frame* prev_frame, String name, int framenum,bool homo=false); //for warped image
void DisplayWeights(frame* image_frame, Mat residual, String name, int framenum ,bool homo=false);
void DisplayOriginalImg(Mat origimg, frame* prev_frame, String name, int framenum,bool homo=false); //for original image
void DisplayColouredDepth(Mat depthimg, Mat img);

//for calculate pixel wise
void DisplayIterationResPixelWise(Mat residualimg,frame* prev_frame, String name, int framenum,bool homo=false);
void DisplayInitialResPixelWise(Mat residualimg,frame* prev_frame, String name, int framenum,bool homo=false);
void DisplayWarpedImgPxelWise(Mat warpedimg,frame* prev_frame, String name, int framenum,bool homo=false);
void DisplayWeightsPixelWise(Mat weights,frame* prev_frame, String name, int framenum,bool homo=false);
void DisplayOriginalImgPixelWise(Mat origimg,frame* prev_frame, String name, int framenum,bool homo=false);


#endif /* defined(__DisplayFunc__) */
