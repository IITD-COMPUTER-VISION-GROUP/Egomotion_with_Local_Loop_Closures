/**
 * This file is a part of Egomotion with Local Loop Closures(ELLC) project.
 * For more information visit: http://www.cse.iitd.ac.in/~suvam/wacv2017.html
 **/

#include "Pyramid.h"
#include "UserDefinedFunc.h"
#include "ExternVariable.h"
#include "DisplayFunc.h"

using namespace cv;

Pyramid::Pyramid(frame* prevframe,frame* currentframe,float* pose,depthMap* currDepthMap){

    currentDepthMap=currDepthMap;
    prev_frame=prevframe;
    current_frame=currentframe;
    
    //initialization of all matrices
    steepestDescent=Mat(prev_frame->no_nonZeroDepthPts,6,CV_32FC1);         //steepest descent matrix (Nx6)
    saveImg= Mat(1, prev_frame->no_nonZeroDepthPts, CV_32FC1);              //save intensity of prev image at non zero depth points for current pyramid level
    worldPoints=Mat(3,prev_frame->no_nonZeroDepthPts,CV_32FC1);             //saves world points (X,Y,Z) in the frame of ref of prev image
    transformedWorldPoints=Mat(3,prev_frame->no_nonZeroDepthPts,CV_32FC1);  //saves world points in the frame of ref of current img
    warpedPoints= Mat(2,prev_frame->no_nonZeroDepthPts, CV_32FC1);          //stores warped points (x,y)
    warpedImage=Mat::zeros(1,prev_frame->no_nonZeroDepthPts,CV_32FC1);
    warpedGradientx=Mat(1,prev_frame->no_nonZeroDepthPts, CV_32FC1);        //gradientx for current img
    warpedGradienty=Mat(1,prev_frame->no_nonZeroDepthPts, CV_32FC1);        //gradienty for current img
    weights=Mat(1,prev_frame->no_nonZeroDepthPts, CV_32FC1);                //weights for residual
   
    //covariance is for implementation of motion prior
    //not being used
    covarianceDiagonalWts[0]=100.0f; //w1
    covarianceDiagonalWts[1]=100.0f;
    covarianceDiagonalWts[2]=100.0f;
    covarianceDiagonalWts[3]=0.01f; //v1
    covarianceDiagonalWts[4]=0.01f;
    covarianceDiagonalWts[5]=0.01f;
    calCovarianceMatrixInv(covarianceDiagonalWts);

}


void Pyramid::calculateSteepestDescent()
{
    //printf("\nCalculating Steepest Descent for frame: %d ", prev_frame->frameId);
  
    //*******INITIALIZE VAR*******//
    
    Mat jacobian_top(prev_frame->no_nonZeroDepthPts,6, CV_32FC1); //top row of jacobian correspoding to x (Nx6)
    Mat jacobian_bottom(prev_frame->no_nonZeroDepthPts,6, CV_32FC1); //bottom row of jacobian corresponding to y (Nx6)
    
    //calculate resized intrinsic focal parameters for pyrlevel
    vector<float> resized_intrinsic= GetIntrinsic(prev_frame->pyrLevel);
    float resized_fx=resized_intrinsic[0];
    float resized_fy=resized_intrinsic[1];
    float resized_cx=resized_intrinsic[2];
    float resized_cy=resized_intrinsic[3];
    
    
    //*******INITIALIZE POINTERS*******//
    
    //pointers to access elements
    float* depth_ptr;
    float* jacob_top_ptr;
    float* jacob_bottom_ptr;

    int jac_rows;
    
    //check if matrix stored continuously
    //affects usage of pointers
    //not usually required to do so
    if(jacobian_top.isContinuous() & jacobian_bottom.isContinuous())
        jac_rows= -1;
    else
        jac_rows= 0;
    
    int jac_counter =0;
    
    jacob_top_ptr = jacobian_top.ptr<float>(0);
    jacob_bottom_ptr = jacobian_bottom.ptr<float>(0);
    
    //*******LOOP TO CALCULATE JACOBIAN*******//
    
    int m; //loop variables
    float i,j;
    int idx=0;
    float* x_warped=warpedPoints.ptr<float>(0);
    float* y_warped=warpedPoints.ptr<float>(1);
    float* gradxwarp=warpedGradientx.ptr<float>(0);
    float* gradywarp=warpedGradienty.ptr<float>(0);
    depth_ptr=transformedWorldPoints.ptr<float>(2); //Z

    float gradx,grady;
    for(m=0;m<prev_frame->no_nonZeroDepthPts;m++) //loop over each pixel with non-zero depth
    {
        
        {
        
            i=y_warped[idx];
            j=x_warped[idx];
            
            if(isinf(i) || isinf(j)) //something wrong if this is true
                printf("\n in calc steepest descent: i: %f, j: %f", i,j);
            
            //calculate gradient at warped points
            gradx=current_frame->getInterpolatedElement(x_warped[idx], y_warped[idx], "gradx");
            grady=current_frame->getInterpolatedElement(x_warped[idx], y_warped[idx], "grady");
            
            gradxwarp[idx]=gradx;
            gradywarp[idx]=grady;
            
            //pre-computed jacobian expression assuming lie algebra pose parameters to be 0
            //matlab proof available
            jacob_bottom_ptr[jac_counter]  = grady*(-(resized_fy + (pow((-resized_cy + i),2) / resized_fy)));
            jacob_top_ptr[jac_counter++] = gradx*(-((-resized_cy + i)*(-resized_cx + j)) / resized_fy);
            
            jacob_bottom_ptr[jac_counter]=grady*(((-resized_cy + i)*(-resized_cx + j)) / resized_fx);
            jacob_top_ptr[jac_counter++] = gradx*(resized_fx + (pow((-resized_cx + j),2) / resized_fx));
            
            jacob_bottom_ptr[jac_counter]=grady*((resized_fy*(-resized_cx + j)) / resized_fx);
            jacob_top_ptr[jac_counter++]  = gradx*(-(resized_fx*(-resized_cy + i) / resized_fy));
            
            jacob_bottom_ptr[jac_counter]=0;
            jacob_top_ptr[jac_counter++]  =gradx*( resized_fx*(pow(depth_ptr[idx],-1)));
            
            jacob_bottom_ptr[jac_counter]= grady*(resized_fy*(pow(depth_ptr[idx],-1)));
            jacob_top_ptr[jac_counter++]  = 0;
            
            jacob_bottom_ptr[jac_counter]= grady*(-(-resized_cy + i) *(pow(depth_ptr[idx],-1)));
            jacob_top_ptr[jac_counter++]  = gradx*(-(-resized_cx + j) *(pow(depth_ptr[idx],-1)));
        }

            if(jac_rows>-1)
            {   jac_rows=jac_rows+1;
                jacob_bottom_ptr=jacobian_bottom.ptr<float>(jac_rows);
                jacob_top_ptr=jacobian_top.ptr<float>(jac_rows);
                jac_counter=0;
            }
            idx++;
        
    }
   
    //*******CALCULATE STEEPEST DESCENT*******//
    steepestDescent=jacobian_top+jacobian_bottom; //jacobian here is already multiplied with gradient
    
    return;
    
}




void Pyramid::calculateHessianInv()
{
    //printf("\nCalculating Hessian ");
    float* weight_ptr=weights.ptr<float>(0);
    
    Mat temp= steepestDescent.t();
    
    weight_ptr=weights.ptr<float>(0);
    float* temp_ptr0=temp.ptr<float>(0);
    float* temp_ptr1=temp.ptr<float>(1);
    float* temp_ptr2=temp.ptr<float>(2);
    float* temp_ptr3=temp.ptr<float>(3);
    float* temp_ptr4=temp.ptr<float>(4);
    float* temp_ptr5=temp.ptr<float>(5);
    
    int y;
    for(y=0; y<prev_frame->no_nonZeroDepthPts; y++)
    {
    
        temp_ptr0[y]=temp_ptr0[y]*weight_ptr[y];
        temp_ptr1[y]=temp_ptr1[y]*weight_ptr[y];
        temp_ptr2[y]=temp_ptr2[y]*weight_ptr[y];
        temp_ptr3[y]=temp_ptr3[y]*weight_ptr[y];
        temp_ptr4[y]=temp_ptr4[y]*weight_ptr[y];
        temp_ptr5[y]=temp_ptr5[y]*weight_ptr[y];
        
        if(isnan(float(weight_ptr[y] )))
        {
            cout<<"\nweight nan: "<<weight_ptr[y];
        }
        
    }
    Mat hessian = Mat::zeros(6, 6, CV_32FC1);
    Mat new_temp=temp.t();
    
    float* new_temp_ptr=new_temp.ptr<float>(0);
    
    //float* temp_ptr = temp.ptr<float>(0);
    float* steep_ptr=steepestDescent.ptr<float>(0);

    for(y=0; y<prev_frame->no_nonZeroDepthPts; y++ )
    {
        new_temp_ptr = new_temp.ptr<float>(y);
        steep_ptr=steepestDescent.ptr<float>(y);
        Mat temp_T= (Mat_<float>(6,1)<<new_temp_ptr[0], new_temp_ptr[1],new_temp_ptr[2], new_temp_ptr[3], new_temp_ptr[4], new_temp_ptr[5] );
        Mat steep_T=(Mat_<float>(1,6)<<steep_ptr[0], steep_ptr[1], steep_ptr[2], steep_ptr[3], steep_ptr[4], steep_ptr[5]);
        hessian+= ((temp_T*steep_T));//+ covarianceMatrixInv);
       // hessian+=covarianceMatrixInv;
    }
   
    hessianInv=hessian.inv();
    
    return;
    
}


//performs inverse projection to give world points in frame of ref of prev image
void Pyramid::calculateWorldPoints()
{
    //printf("\nCalculating World Points for previous image: %d", prev_frame->frameId);
    int pyrlevel=prev_frame->pyrLevel;
    int nRows=prev_frame->currentRows;
    int nCols=prev_frame->currentCols;
    //*******INITIALIZE POINTERS*******//
    
    //worldpoint pointers
    float* worldpoint_ptr0; //pointer to access row 0
    float* worldpoint_ptr1; //pointer to access row 1
    float* worldpoint_ptr2; //pointer to access row 2
    uchar* img_ptr;
    float* saveimg_ptr;
    uchar* mask_ptr;
    float* depth_ptr;
    
    worldpoint_ptr0=worldPoints.ptr<float>(0); //initialize to row0
    worldpoint_ptr1=worldPoints.ptr<float>(1); //initialize to row1
    worldpoint_ptr2=worldPoints.ptr<float>(2); //initialize to row2
    
    saveimg_ptr=saveImg.ptr<float>(0); //initialize to row 0 (single row Mat)
    
    //*******CALCULATE RESIZED INTRINSIC PARAMETERS*******//
    
    vector<float> resized_intrinsic= GetIntrinsic(prev_frame->pyrLevel);
    float resized_fx=resized_intrinsic[0];
    float resized_fy=resized_intrinsic[1];
    float resized_cx=resized_intrinsic[2];
    float resized_cy=resized_intrinsic[3];
    
    //*******INITIALIZE COUNTERS*******//
    
    int world_rows0=0; //counter to access columns in row 0
    int world_rows1=0; //counter to access columns in row 1
    int world_rows2=0; //counter to access columns in row 2
    int saveimg_row=0; //counter to store in ro0 of saveimg
    
    int i,j;//loop variables
    for(i = 0; i < nRows; ++i)
    {
        //get pointer for current row i;
        img_ptr = prev_frame->image_pyramid[pyrlevel].ptr<uchar>(i);
        mask_ptr=prev_frame->mask.ptr<uchar>(i);
        depth_ptr = prev_frame->depth_pyramid[pyrlevel].ptr<float>(i);
        
        for (j = 0; j < nCols; ++j)        
        {   //check for zero depth point
            if(mask_ptr[j]==0)
                continue;  //skip loop for non zero depth points
            
            //save image intensity at non zero depth point
            saveimg_ptr[saveimg_row++]=img_ptr[j];
            
            //populating world point with homogeneous coordinate ( X ;Y ;depth ; 1)
            worldpoint_ptr0[world_rows0++]=(j-resized_cx)*depth_ptr[j]/resized_fx; //X-coordinate
            worldpoint_ptr1[world_rows1++]=(i-resized_cy)*depth_ptr[j]/resized_fy; //Y-coordinate
            worldpoint_ptr2[world_rows2++]=depth_ptr[j]; //depth of point at (u,v)
            
        }
    }
    return;
}

//calculates warped pixel coordinates in the current image
void Pyramid::calculateWarpedPoints ()
{
    //printf("\nCalculating warped points for previous frame: %d ", prev_frame->frameId);
    //*******CALCULATE POSE MATRIX *******//
    
    int nonZeroPts=prev_frame->no_nonZeroDepthPts;
    
    //Create matrix in OpenCV
    Mat se3=(Mat_<float>(4, 4) << 0,-pose[2],pose[1],pose[3], pose[2],0,-pose[0],pose[4], -pose[1],pose[0],0,pose[5],0,0,0,0);
    
    // Map the OpenCV matrix with Eigen:
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> se3_Eigen(se3.ptr<float>(), se3.rows, se3.cols);
    
    // Take exp in Eigen and store in new Eigen matrix
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> SE3_Eigen = se3_Eigen.exp();
    
    // create an OpenCV Mat header for the Eigen data:
    Mat SE3(4, 4, CV_32FC1, SE3_Eigen.data()); //SE3 pose of current w.r.t prev image
    
    //*******INITIALIZE POINTERS*******//
    
    //warpedpoint pointers
    float* warpedpoint_ptrx; //pointer to access row 0=> x image coordinate
    float* warpedpoint_ptry; //pointer to access row 1=> y image coordinate
    //worldpoint pointers
    float* worldpoint_ptrX; //pointer to access row 0=> X
    float* worldpoint_ptrY; //pointer to access row 1=> Y
    float* worldpoint_ptrZ; //pointer to access row 2=> Z
    float* trfm_worldpoint_ptrX;
    float* trfm_worldpoint_ptrY;
    float* trfm_worldpoint_ptrZ;

    
    warpedpoint_ptrx=warpedPoints.ptr<float>(0); //initialize to row0
    warpedpoint_ptry=warpedPoints.ptr<float>(1); //initialize to row1
    
    worldpoint_ptrX=worldPoints.ptr<float>(0); //initialize to row0
    worldpoint_ptrY=worldPoints.ptr<float>(1); //initialize to row1
    worldpoint_ptrZ=worldPoints.ptr<float>(2); //initialize to row2
    
    trfm_worldpoint_ptrX=transformedWorldPoints.ptr<float>(0); //initialize to row0
    trfm_worldpoint_ptrY=transformedWorldPoints.ptr<float>(1); //initialize to row1
    trfm_worldpoint_ptrZ=transformedWorldPoints.ptr<float>(2); //initialize to row2

    
    //*******CALCULATE RESIZED INTRINSIC PARAMETERS*******//
    
    vector<float> resized_intrinsic= GetIntrinsic(prev_frame->pyrLevel);
    float resized_fx=resized_intrinsic[0];
    float resized_fy=resized_intrinsic[1];
    float resized_cx=resized_intrinsic[2];
    float resized_cy=resized_intrinsic[3];
    
    //*******STORE SE3 PARAMETERS*******//
    
    float* SE3_ptr;
    float SE3_vec[12];  //r11, r12, r13, t1, r21 r22, r23, t2, r31, r32, r33, t3
    int vec_counter=0;
    int i; //loop variables
    
    for(i=0; i<3; ++i)
    {
        SE3_ptr=SE3.ptr<float>(i); //get pointer to first row of SE3
        
        SE3_vec[vec_counter++]=SE3_ptr[0];
        SE3_vec[vec_counter++]=SE3_ptr[1];
        SE3_vec[vec_counter++]=SE3_ptr[2];
        SE3_vec[vec_counter++]=SE3_ptr[3];    
    }
    
    //enter loop to calculate warped points
    int j;  
    for( j = 0; j < nonZeroPts; ++j)
    {  
        if (SE3_vec[1]==0) //if else condition here may not be needed
        {
            trfm_worldpoint_ptrX[j]=((SE3_vec[0]*worldpoint_ptrX[j])+(SE3_vec[1]*worldpoint_ptrY[j])+(SE3_vec[2]*worldpoint_ptrZ[j])+(SE3_vec[3]));
            trfm_worldpoint_ptrY[j]=((SE3_vec[4]*worldpoint_ptrX[j])+(SE3_vec[5]*worldpoint_ptrY[j])+(SE3_vec[6]*worldpoint_ptrZ[j])+(SE3_vec[7]));
            trfm_worldpoint_ptrZ[j]=((SE3_vec[8]*worldpoint_ptrX[j])+(SE3_vec[9]*worldpoint_ptrY[j])+(SE3_vec[10]*worldpoint_ptrZ[j])+(SE3_vec[11]));
            
            trfm_worldpoint_ptrZ[j]=UNZERO(trfm_worldpoint_ptrZ[j]); //if depth becomes zero
            
            warpedpoint_ptrx[j]=((trfm_worldpoint_ptrX[j]/trfm_worldpoint_ptrZ[j])*resized_fx)+resized_cx;
            
            warpedpoint_ptry[j]=((trfm_worldpoint_ptrY[j]/trfm_worldpoint_ptrZ[j])*resized_fy)+resized_cy;          
        }
        
        else
        {
            trfm_worldpoint_ptrX[j]=(float(SE3_vec[0]*worldpoint_ptrX[j])+float(SE3_vec[1]*worldpoint_ptrY[j])+float(SE3_vec[2]*worldpoint_ptrZ[j])+float(SE3_vec[3]));
            trfm_worldpoint_ptrY[j]=(float((SE3_vec[4]*worldpoint_ptrX[j]))+float((SE3_vec[5]*worldpoint_ptrY[j]))+float((SE3_vec[6]*worldpoint_ptrZ[j]))+float((SE3_vec[7])));
            trfm_worldpoint_ptrZ[j]=(float(SE3_vec[8]*worldpoint_ptrX[j])+float(SE3_vec[9]*worldpoint_ptrY[j])+float(SE3_vec[10]*worldpoint_ptrZ[j])+float(SE3_vec[11]));
            
            trfm_worldpoint_ptrZ[j]=UNZERO(trfm_worldpoint_ptrZ[j]); //if depth becomes zero
            
            warpedpoint_ptrx[j]=float(((trfm_worldpoint_ptrX[j]/trfm_worldpoint_ptrZ[j])*resized_fx)+resized_cx);
            
            warpedpoint_ptry[j]=float(((trfm_worldpoint_ptrY[j]/trfm_worldpoint_ptrZ[j])*resized_fy)+resized_cy);
            

            //debugging print statements
            if(isinf(warpedpoint_ptrx[j]) || isinf(warpedpoint_ptry[j]))
            {
                printf("\n In warped points: warpedx: %f, warpedy: %f", warpedpoint_ptrx[j], warpedpoint_ptry[j]);
                printf("\n\ntrfm_worldpointx: %f, trfm_worldpointy: %f, trfm_worldpointz: %f, worldpointx: %f, worldpointy: %f, worldpointz: %f", trfm_worldpoint_ptrX[j],  trfm_worldpoint_ptrY[j],  trfm_worldpoint_ptrZ[j],worldpoint_ptrX[j], worldpoint_ptrY[j], worldpoint_ptrZ[j]  );
                
                printf("\nSE3_vec: %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f", SE3_vec[0],  SE3_vec[1],  SE3_vec[2],  SE3_vec[3],  SE3_vec[4],  SE3_vec[5],  SE3_vec[6],  SE3_vec[7],  SE3_vec[8],  SE3_vec[9],  SE3_vec[10],  SE3_vec[11]);
                
                printf("\npose: %f, %f, %f, %f, %f, %f", pose[0],pose[1], pose[2], pose[3], pose[4], pose[5] );
                
                cout<<"\nterm1: "<<SE3_vec[8]*worldpoint_ptrX[j];
                cout<<"\nterm2: "<<SE3_vec[9]*worldpoint_ptrY[j];
                cout<<"\nterm3: "<<SE3_vec[10]*worldpoint_ptrZ[j];
                cout<<"\nterm4: "<<SE3_vec[11];
                
                printf("\nstopping...");
                
            }
 
        }   
    } 
    
    return;
}


//calculate intensity of warped points in current img to give warped image
//intensity evaluates as -1 if warped point is out of bounds(oob)
void Pyramid::calculateWarpedImage()
{
    //printf("\nCalculating Warped Image for previous frame: %d, current frame: %d", prev_frame->frameId, current_frame->frameId);
    int nonZeroPts=prev_frame->no_nonZeroDepthPts;
    int pyrlevel=prev_frame->pyrLevel;
    
    
    //*******INITIALIZE POINTERS*******//
    
    //initialize pointers
    uchar* img_ptr0; //pointer0 to access original image
    uchar* img_ptr1; //pointer1 to acess original image
    float* warpedimg_ptr; //pointer to store in warped image
    float* warpedpoint_ptrx; //pointer to acess warped point x => column
    float* warpedpoint_ptry; //pointer to access warped point y=> row
    
    warpedpoint_ptrx=warpedPoints.ptr<float>(0); //initialize to row0 to acess x coordinate
    warpedpoint_ptry=warpedPoints.ptr<float>(1); //initialize to row1 to acess y coordinate
    warpedimg_ptr=warpedImage.ptr<float>(0); //initialize to store in row0 of warped image
    
    
    //*******DECLARE VARIABLES*******//
    
    //cout<<"\n\nWARPED POINTS   "<<warpedpoint;
    
    
    int j; //for loop variables
    float yx[2]; //to store warped points for current itertion
    float wt[2]; //to store weight
    float y,x; //to store x and y coordinate to acess original image
    uchar pixVal1, pixVal2; //to store intensity value
    float interTop, interBtm;
    
    int nCols=prev_frame->currentCols-1; //maximum valid x coordinate
    int nRows=prev_frame->currentRows-1; //maximum valid y coordinate
    
    for(j=0; j<nonZeroPts; j++ )
    {
        int countOutOfBounds=0;
        yx[0]=warpedpoint_ptry[j]; //store current warped point y
        yx[1]=warpedpoint_ptrx[j]; //store warped point x
        
        
        wt[0]=yx[0]-floor(yx[0]); //weight for y
        wt[1]=yx[1]-floor(yx[1]); //weight for x
        
        //Case 1
        y=floor(yx[0]); //floor of y
        x=floor(yx[1]); //floor of x
        
        //cout<<"\n\n"<<int(x);
        if ((x<0)||(x>nCols)||(y<0)||(y>nRows))
        {   
            pixVal1=0; //if outside boundary pixel value is 0
            countOutOfBounds++;
        }
        else
        {
            img_ptr0=current_frame->image_pyramid[pyrlevel].ptr<uchar>(y); //initialize image pointer0 to row floor(y)
            pixVal1=img_ptr0[int(x)]; //move pointer to get value at pixel (floor(x), floor(y))
        }
        

        x=yx[1]; //warped point x
        
        if ((x<0)||(x>nCols)||(y<0)||(y>nRows))
        {
            pixVal2=0; //if outside boundary pixel value is 0
            countOutOfBounds++;
        }
        else
        {
            img_ptr0=current_frame->image_pyramid[pyrlevel].ptr<uchar>(y); //initialize image pointer0 to row floor(y)
            pixVal2=img_ptr0[int(ceil(x))]; //move pointer to get value at pixel (ceil(x), floor(y))
        }
        
        interTop=((1-wt[1])*pixVal1)+(wt[1]*pixVal2);
        
        
        //Case 2
        y=yx[0]; //warped point y
        
        x=floor(yx[1]); //floor of x
        
        if ((x<0)||(x>nCols)||(y<0)||(y>nRows))
        {
            pixVal1=0; //if outside boundary pixel value is 0
            countOutOfBounds++;
        }
        else
        {
            img_ptr1=current_frame->image_pyramid[pyrlevel].ptr<uchar>(ceil(y)); //initialize image pointer1 to row ceil(y)
            pixVal1=img_ptr1[int(x)]; //move pointer to get value at pixel (floor(x), ceil(y))
        }
        
        x=yx[1]; //warped point x
        
        if ((x<0)||(x>nCols)||(y<0)||(y>nRows))
        {   
            pixVal2=0; //if outside boundary pixel value is 0
            countOutOfBounds++;
        }
        else
        {
            img_ptr1=current_frame->image_pyramid[pyrlevel].ptr<uchar>(ceil(y)); //initialize image pointer1 to row ceil(y)
            pixVal2=img_ptr1[int(ceil(x))]; //move pointer to get value at pixel (ceil(x), ceil(y))
        }
        
        if(countOutOfBounds==4)
        {
            warpedimg_ptr[j]=-1.0f;
        }
        else
        {
            interBtm=((1-wt[1])*pixVal1)+(wt[1]*pixVal2);
            warpedimg_ptr[j]=((1-wt[0])*interTop)+(wt[0]*interBtm); //calculate interpolated value to get warped image intensity
        }     
    }
    
    return;
    
}

//calculates delta pose (pose change) and updates current estimate of pose
void Pyramid::updatePose()
{
    //printf("\nUpdating pose ");
    Mat temp=residual.mul(weights);

    Mat sd_param= temp*steepestDescent; //calculate steepest descent parameters (sdparams)

    
    Mat deltapose1= ((hessianInv*(sd_param.t())).t()); //calculate delta pose (change in pose)
    deltapose1=-deltapose1; //negating here as residual=(warpedImage-previmage), if residual=(previmage-warpedImage) then no minus sign needed

    Mat deltapose2=(hessianInv*(motionPrior.t())).t(); //motion prior term, not used
    
    deltapose=deltapose1; //final deltapose
    
    float* deltapose_ptr=deltapose.ptr<float>(0); //initialize pointer to delta pose
    

    //calculate weighted pose value
    float weighted_pose = abs(deltapose_ptr[0]*util::weight[0])+abs(deltapose_ptr[1]*util::weight[1])+abs(deltapose_ptr[2]*util::weight[2])+abs(deltapose_ptr[3]*util::weight[3])+abs(deltapose_ptr[4]*util::weight[4])+abs(deltapose_ptr[5]*util::weight[5]);
    
    weightedPose=weighted_pose; //only for the purpose of checking termination condition, not needed in pose est.
 
    current_frame->concatenateRelativePose(deltapose_ptr, pose, pose); //estimate of pose updated
    
}



//calculates residual image after warping and pixel weights
float Pyramid::calResidualAndWeights()
{
    //printf("\ncalculating weights and residual");
    vector<float> resized_intrinsic= GetIntrinsic(prev_frame->pyrLevel);
    float resized_fx=resized_intrinsic[0];
    float resized_fy=resized_intrinsic[1];

    
    float weightedSumOfResidual=0.0f;
    residual=(warpedImage-saveImg); //residual image=warpedimg-previmg , calculated only at non-zero depth points
    
    Mat maskOutOfBounds=warpedImage!=-1.0f; //where warped intensity is -1, signifies oob pixels
    //cout<<"\n\n"<<maskOutOfBounds;
    
    uchar* mask_ptr=maskOutOfBounds.ptr<uchar>(0);
    float* residual_ptr=residual.ptr<float>(0);
    
    int j;
    for(j=0 ; j<maskOutOfBounds.cols ; j++)
    {
        if(mask_ptr[j]==0)
            residual_ptr[j]=0.0f;// at oob pixels, residual=0
    }

    
    float* trfmpoint_x=transformedWorldPoints.ptr<float>(0);
    float* trfmpoint_y=transformedWorldPoints.ptr<float>(1);
    float* trfmpoint_z=transformedWorldPoints.ptr<float>(2);
    
    uchar* maskptr=prev_frame->mask.ptr<uchar>(0);
    
    float* weightptr=weights.ptr<float>(0);
    float* residualptr=residual.ptr<float>(0);
    float* depth_ptr=prev_frame->depth_pyramid[prev_frame->pyrLevel].ptr<float>(0);
    
    float* gradxwarp=warpedGradientx.ptr<float>(0);
    float* gradywarp=warpedGradienty.ptr<float>(0);
    
    int idx=0;
   
    Mat se3=(Mat_<float>(4, 4) << 0,-pose[2],pose[1],pose[3], pose[2],0,-pose[0],pose[4], -pose[1],pose[0],0,pose[5],0,0,0,0);
    
    // Map the OpenCV matrix with Eigen:
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> se3_Eigen(se3.ptr<float>(), se3.rows, se3.cols);
    
    // Take exp in Eigen and store in new Eigen matrix
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> SE3_Eigen = se3_Eigen.exp();
    
    float tx=SE3_Eigen(0,3);
    float ty=SE3_Eigen(1,3);
    float tz=SE3_Eigen(2,3);
    
    int width= prev_frame->currentCols;
    
    float usageCount=0;
    
    
    //calculating pixel weights, depends on residual and depth variance at that point
    int y,x;
    for(y=0;y<prev_frame->currentRows;y++)
    {
        maskptr=prev_frame->mask.ptr<uchar>(y);
        depth_ptr=prev_frame->depth_pyramid[prev_frame->pyrLevel].ptr<float>(y);
        
        for (x=0; x<prev_frame->currentCols; x++)
        {
            if(maskptr[x]==0)
            {   //weightptr[idx]=1;
                continue;
            }
            
            float px = trfmpoint_x[idx];	// x'
            float py = trfmpoint_y[idx];	// y'
            float pz = trfmpoint_z[idx];	// z'
            float d = 1.0f/depth_ptr[x];	// d
            float rp = residualptr[idx]; // r_p
            float gx = resized_fx * gradxwarp[idx];	// \delta_x I
            float gy = resized_fy * gradywarp[idx];  // \delta_y I
            float s = 1.0f * (*(currentDepthMap->depthvararrptr[prev_frame->pyrLevel]+x+width*y));	// \sigma_d^2
            // calc dw/dd (first 2 components):
            
            float g0 = (tx * pz - tz * px) / (pz*pz*d);
            float g1 = (ty * pz - tz * py) / (pz*pz*d);
            
            // calc w_p
            float drpdd = gx * g0 + gy * g1;	// ommitting the minus
            float w_p = 1.0f / (util::CAMERA_PIXEL_NOISE_2 + s * drpdd * drpdd);
            float weighted_rp = fabs(rp*sqrtf(w_p));
            
            float wh = fabs(weighted_rp < (util::HUBER_D/2) ? 1 : (util::HUBER_D/2) / weighted_rp);
            
            //sumRes += wh * w_p * rp*rp;
            
            weightptr[idx] = wh * w_p; //final weight of pixel, weight matrix populated here
            
            
            //for debugging
            if (isnan(weightptr[idx]))
            {
                printf("\n\nWeights:%f , wh: %f, wp: %f, weighted_rp: %f, rp: %f, residualptr[idx]: %f, s: %f, drpdd: %f; ",weightptr[idx], wh, w_p, weighted_rp, rp, residualptr[idx], s, drpdd);
                printf("\n\npx: %f, py: %f, pz: %f, d: %f, gx: %f, gy: %f, g0: %f, g1: %f, gradxwarp: %f, gradywarp: %f, idx: %d", px, py, pz, d, gx, gy, g0, g1, gradxwarp[idx], gradywarp[idx], idx);
                cout<<"\n\nwarpedGradientx: \n"<<warpedGradientx;
                cout<<"\n\nwarpedGradienty: \n"<<warpedGradienty;
                
                float* new_grad_ptrx=warpedGradientx.ptr<float>(0);
                float* new_grad_ptry=warpedGradienty.ptr<float>(0);
                //cout<<"\n\n\n\nNew grad pointer x: "<<new_grad_ptrx[idx]<<"New grad pointer y: "<<new_grad_ptry[idx];
                //cout<<"\nprev non-zero: "<<prev_frame->no_nonZeroDepthPts;
                cout<<"\nlevel: "<<prev_frame->currentCols;
     
                int k;
                for(k=0; k<prev_frame->no_nonZeroDepthPts; k++)
                {
                    if(isnan(new_grad_ptrx[k]) || isnan(new_grad_ptry[k]))
                        cout<<"\n k: "<<k<<" -> "<<new_grad_ptrx[k]<<" , "<<new_grad_ptry[k];
                }
            }
            
            //for debugging
            if(w_p < 0)
            {    printf("\n\n!!!!!!!!!w_p: %f, x =%d, y=%d",w_p, x, y);
               // waitKey(0);
            }
            
            weightedSumOfResidual+=weightptr[idx]*rp*rp;
            
            idx++;
            float depthChange = 1.0f/ d*pz;	// if depth becomes larger: pixel becomes "smaller", hence count it less.
            usageCount += depthChange < 1 ? depthChange : 1; //not used in this implementation
            
        }

    }
    pointUsage=usageCount/float(prev_frame->no_nonZeroDepthPts); //not used
    
    return weightedSumOfResidual/float(prev_frame->no_nonZeroDepthPts);
}


//performed before each iteration
//not precomputation in exact sense as many steps are repeated in the iteration step
//this is because it is an implementation of the lucas kanade compositional algorithm
void Pyramid::performPrecomputation(){
    
    //printf("\nPerforming pre-computation..");
    
    calculateWorldPoints();
    calculateWarpedPoints();
    calculateWarpedImage();
    calculateSteepestDescent();
    lastErr=calResidualAndWeights();

    return;
}

//this is called in each iteration at a pyrimid level
float Pyramid::performIterationSteps(){

    calculateHessianInv();
    calMotionPrior();
    updatePose();
    calculateWarpedPoints();
    calculateWarpedImage();
    calculateSteepestDescent();
    error=calResidualAndWeights();
    float temp=lastErr;
    lastErr=error;
    return error/temp;
}

void Pyramid::putPreviousPose(frame* tminus1_prev_frame)
{
    //prevPose=tminus1_prev_frame->poseWrtOrigin;
    prevPose[0]=tminus1_prev_frame->poseWrtOrigin[0];
    prevPose[1]=tminus1_prev_frame->poseWrtOrigin[1];
    prevPose[2]=tminus1_prev_frame->poseWrtOrigin[2];
    prevPose[3]=tminus1_prev_frame->poseWrtOrigin[3];
    prevPose[4]=tminus1_prev_frame->poseWrtOrigin[4];
    prevPose[5]=tminus1_prev_frame->poseWrtOrigin[5];

}

//part of motion prior, not used
void Pyramid::calCovarianceMatrixInv(float* covar_wts)
{
    Mat covarianceMatrix=Mat::zeros(6, 6, CV_32FC1);
    
    float* covarmat_ptr=covarianceMatrix.ptr<float>(0);

    int y;
    for(y=0;y<6;y++)
    {
        covarmat_ptr=covarianceMatrix.ptr<float>(y);
        covarmat_ptr[y]=covar_wts[y];
    }

    covarianceMatrixInv=covarianceMatrix.inv();
}

//part of motion prior, not used 
void Pyramid::calMotionPrior()
{
    //printf("\nIn motion prior");
    Mat diffPose=Mat::zeros(1, 6, CV_32FC1);
    float* diffpose_ptr=diffPose.ptr<float>(0);
    
    int i;
    for(i=0;i<6; i++)
    {
        diffpose_ptr[i]=prevPose[i]-pose[i];
    }
    motionPrior=Mat::zeros(1, 6, CV_32FC1);
    
    for(i=0;i<prev_frame->no_nonZeroDepthPts; i++)
        motionPrior+=(covarianceMatrixInv*(diffPose.t())).t(); //1x6
    
}
                   
