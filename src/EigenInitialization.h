#ifndef __EigenInitialization__
#define __EigenInitialization__

/**
 * This file is a part of Egomotion with Local Loop Closures(ELLC) project.
 * For more information visit: http://www.cse.iitd.ac.in/~suvam/wacv2017.html
 **/

#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <unsupported/Eigen/MatrixFunctions>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include "opencv2/opencv.hpp"
//#include "ExternVariable.h"

using namespace cv;
using namespace Eigen;
using namespace std;


namespace util
{
    
    extern Mat K;
    extern Mat Kinv;
    
    extern Map<Matrix<float, Dynamic, Dynamic, RowMajor>> K_Eigen;
    extern Map<Matrix<float, Dynamic, Dynamic, RowMajor>> Kinv_Eigen;
        
      
    extern float* Kinv_ptr;
    extern float ORIG_FX_INV ;
    extern float ORIG_CX_INV ;
    
    extern float* Kinv_ptr2;
    extern float ORIG_FY_INV ;
    extern float ORIG_CY_INV ;

}


#endif /* defined(__EigenInitialization__) */



