
/**
 * This file is a part of Egomotion with Local Loop Closures(ELLC) project.
 * For more information visit: http://www.cse.iitd.ac.in/~suvam/wacv2017.html
 **/

#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <unsupported/Eigen/MatrixFunctions>

#include <cstdio>
#include <cstdlib>
#include "opencv2/opencv.hpp"
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include "ExternVariable.h"

#include "UserDefinedFunc.h"
#include "DisplayFunc.h"


using namespace std;
using namespace cv;


Mat FormImageMask(Mat depth_img)
{
    //PRINTF("\nCalculating Image Mask");
    return (depth_img>0.0f);
}


//returns resized intrinsic matrx [fx,fy,cx,cy]
vector<float> GetIntrinsic(int pyrlevel)
{
    //PRINTF("\nCalculating Intrinsic parameters for level: %d", pyrlevel);
    vector<float> resized_intrinsic(4);

    //calculating resized intrinsic parameters  fx, fy, cx, cy
    resized_intrinsic[0] = (util::ORIG_FX / pow(2, pyrlevel));
    resized_intrinsic[1] = (util::ORIG_FY / pow(2, pyrlevel));
    resized_intrinsic[2] = (util::ORIG_CX / pow(2, pyrlevel));
    resized_intrinsic[3] = (util::ORIG_CY / pow(2, pyrlevel));
    
    //storing in resized_intrinsic=[fx, fy, cx, cy];
    //Mat resized_intrinsic = (Mat_<float>(1,4)<<resized_fx, resized_fy, resized_cx, resized_cy);
    
    //return pointer to the first element  
    return resized_intrinsic;
}



