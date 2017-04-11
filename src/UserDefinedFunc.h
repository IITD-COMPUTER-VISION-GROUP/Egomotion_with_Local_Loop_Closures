/**
 * This file is a part of Egomotion with Local Loop Closures(ELLC) project.
 * For more information visit: http://www.cse.iitd.ac.in/~suvam/wacv2017.html
 **/

#pragma once

#ifndef __UserDefinedFunc__
#define __UserDefinedFunc__

#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <unsupported/Eigen/MatrixFunctions>

#include <cstdio>
#include <cstdlib>
#include "opencv2/opencv.hpp"
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include "ExternVariable.h"
#include "Frame.h"

using namespace std;
using namespace cv;


//form image mask
Mat FormImageMask(Mat depth_img);

//returns intrinsic focal parameters
vector<float> GetIntrinsic(int pyrlevel);

#endif /* defined(__UserDefinedFunc__) */
