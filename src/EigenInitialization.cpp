/**
 * This file is a part of Egomotion with Local Loop Closures(ELLC) project.
 * For more information visit: http://www.cse.iitd.ac.in/~suvam/wacv2017.html
 **/

#include "EigenInitialization.h"
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <unsupported/Eigen/MatrixFunctions>
#include "ExternVariable.h"

using namespace cv;
using namespace Eigen;
using namespace std;


namespace util
{
	    
	Mat K=(Mat_<float>(3,3)<<ORIG_FX,0,ORIG_CX,0,ORIG_FY,ORIG_CY,0,0,1);
	Mat Kinv=K.inv();

	Map<Matrix<float, Dynamic, Dynamic, RowMajor>> K_Eigen(K.ptr<float>(), K.rows, K.cols);
	Map<Matrix<float, Dynamic, Dynamic, RowMajor>> Kinv_Eigen(Kinv.ptr<float>(), Kinv.rows, Kinv.cols);

	float* Kinv_ptr=Kinv.ptr<float>(0);

	float ORIG_FX_INV = Kinv_ptr[0];
	float ORIG_CX_INV = Kinv_ptr[2];

	float* Kinv_ptr2=Kinv.ptr<float>(1);
	
	float ORIG_FY_INV = Kinv_ptr2[1];
	float ORIG_CY_INV = Kinv_ptr2[2];
    
}
