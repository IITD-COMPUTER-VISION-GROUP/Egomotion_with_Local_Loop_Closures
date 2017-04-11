/**
 * This file is a part of Egomotion with Local Loop Closures(ELLC) project.
 * For more information visit: http://www.cse.iitd.ac.in/~suvam/wacv2017.html
 **/

#include <cstdio>
#include <cstdlib>
#include "opencv2/opencv.hpp"
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include "UserDefinedFunc.h"
#include <string>

#include "DisplayFunc.h"
using namespace std;
using namespace cv;


void DEBUG_COND_FUNC(bool cond)
{
    if (cond)
        printf("debug condition hit\n");
}

void MY_ASSERT_FUNC(bool cond)
{
    if (!cond)
    {
        printf("\n\n assertion failure\n");
        exit(1);
    }    
}

void DisplayIterationRes(frame* image_frame, Mat residual,String name, int framenum,bool homo )
{
    PRINTF("\nDisplaying Iteraion residual");
    int nRows = image_frame->currentRows;
    int nCols = image_frame->currentCols;
    
    Mat displayimg = image_frame->mask.clone();
    
    uchar* displayimg_ptr;
    float* residual_ptr;
    int residual_counter=0;
    
    residual_ptr=residual.ptr<float>(residual_counter);

    int i,j;
    for(i=0;i<nRows;++i)
    {
        displayimg_ptr=displayimg.ptr<uchar>(i); //initialize to row 0 (single row Mat)
        
        for(j=0;j<nCols;++j)
        {
            if(displayimg_ptr[j]==0)
                continue;
            
            else
                displayimg_ptr[j]=uchar((abs(residual_ptr[residual_counter++])));
            
        }
    }

    //namedWindow( name, WINDOW_AUTOSIZE );// Create a window for display.
	DEBUG_STATEMENT(imshow(name, displayimg));
    
    //save image
    stringstream ss;
    string str = "final_res_";
    string type = ".jpg";
    
    ss<<"Test_images/"<<str<<(framenum-1)<<"_"<<int(homo)<<type;
    string filename = ss.str();
    ss.str("");
    
    imwrite(filename, displayimg);
    
    return;
};




void DisplayInitialRes(frame* currentFrame,frame* prev_frame,String name, int framenum,bool homo )
{
    PRINTF("\nDisplaying Initial Residual");
    int nRows=prev_frame->height;
    int nCols=prev_frame->width;
    
    Mat maskimg(util::ORIG_ROWS,util::ORIG_COLS,CV_8UC1);//stores depth mask used to display original image residual
    maskimg=FormImageMask(prev_frame->depth); //original resolution depth
    
    Mat residual(util::ORIG_ROWS,util::ORIG_COLS,CV_8UC1);
    residual=abs(currentFrame->image-prev_frame->image);
    
    Mat displayimg=maskimg.clone();
    
    uchar* displayimg_ptr;
    uchar* residual_ptr;

    
    
    int i,j;
    for(i=0;i<nRows;++i)
    {
        displayimg_ptr=displayimg.ptr<uchar>(i); //initialize to row 0 (single row Mat)
        residual_ptr=residual.ptr<uchar>(i);
        
        for(j=0;j<nCols;++j)
        {
            if(displayimg_ptr[j]==0)
                continue;
            
            else
                displayimg_ptr[j]=uchar((abs(residual_ptr[j])));
        }
        
    }
    //namedWindow( name, WINDOW_AUTOSIZE );// Create a window for display.
	DEBUG_STATEMENT(imshow(name, displayimg));
    
    //save image
    stringstream ss;
    string str = "initial_res_";
    string type = ".jpg";
    
    ss<<"Test_images/"<<str<<(framenum-1)<<"_"<<int(homo)<<type;
    string filename = ss.str();
    ss.str("");
    
    imwrite(filename, displayimg);
    
    
    return;
};




void DisplayWarpedImg(Mat warpedimg,frame* prev_frame, String name, int framenum,bool homo)
{
    PRINTF("\nDisplaying Warped image");
    Mat displayimg=prev_frame->mask.clone();
    int nRows=prev_frame->currentRows;
    int nCols=prev_frame->currentCols;
    
    uchar* displayimg_ptr;
    float* warpedimg_ptr;
    int warpedimg_counter=0;
    
    warpedimg_ptr=warpedimg.ptr<float>(warpedimg_counter);
    
    
    int i,j;
    for(i=0;i<nRows;++i)
    {
        displayimg_ptr=displayimg.ptr<uchar>(i); //initialize to row 0 (single row Mat)
       // warpedimg_ptr=warpedimg.ptr<uchar>(i);
        
        for(j=0;j<nCols;++j)
        {
            if(displayimg_ptr[j]==0)
                continue;
            
            else
                displayimg_ptr[j]=uchar((abs(warpedimg_ptr[warpedimg_counter++])));
        }
        
    }
    //namedWindow( name, WINDOW_AUTOSIZE );// Create a window for display.
	DEBUG_STATEMENT(imshow(name, displayimg));
    
    //save image
    stringstream ss;
    string str = "warped_img_";
    string type = ".jpg";
    
    ss<<"Test_images/"<<str<<(framenum-1)<<"_"<<int(homo)<<type;
    string filename = ss.str();
    ss.str("");
    
    imwrite(filename, displayimg);
    
    return;
    
}





void DisplayOriginalImg(Mat origimg, frame* prev_frame, String name, int framenum,bool homo)//for original image
{
    PRINTF("\nDisplaying Original image");
        
        Mat displayimg=prev_frame->mask.clone();
    

        int nRows=prev_frame->currentRows;
        int nCols=prev_frame->currentCols;
    
    
        uchar* displayimg_ptr;
        float* origimg_ptr;
        int origimg_counter=0;
        
        origimg_ptr=origimg.ptr<float>(origimg_counter);
        
    
        int i,j;
        for(i=0;i<nRows;++i)
        {
            displayimg_ptr=displayimg.ptr<uchar>(i); //initialize to row 0 (single row Mat)
            
            for(j=0;j<nCols;++j)
            {
                if(displayimg_ptr[j]==0)
                    continue;
                
                else
                    displayimg_ptr[j]=uchar((abs(origimg_ptr[origimg_counter++])));
                    }
            
        }
        //namedWindow( name, WINDOW_AUTOSIZE );// Create a window for display.
		DEBUG_STATEMENT(imshow(name, displayimg));
        
        //save image
        stringstream ss;
        string str = "orig_img_";
        string type = ".jpg";
    
        ss<<"Test_images/"<<str<<(framenum-1)<<type;
        string filename = ss.str();
        ss.str("");
        
        imwrite(filename, displayimg);
        
        //waitKey(0);
        
        return;
}


void DisplayWeights(frame* image_frame, Mat weights,String name, int framenum,bool homo )
{
    PRINTF("\nDisplaying Weights");
    int nRows=image_frame->currentRows;
    int nCols=image_frame->currentCols;
    
    Mat displayimg=image_frame->mask.clone();
    
    uchar* displayimg_ptr;
    float* weights_ptr;
    int residual_counter=0;
    
    weights_ptr=weights.ptr<float>(residual_counter);
    
    
    int i,j;
    for(i=0;i<nRows;++i)
    {
        displayimg_ptr=displayimg.ptr<uchar>(i); //initialize to row 0 (single row Mat)
        
        for(j=0;j<nCols;++j)
        {
            if(displayimg_ptr[j]==0)
                continue;
            
            else
                displayimg_ptr[j]=uchar((abs(weights_ptr[residual_counter++])*4000));
            
        }
        
    }
    // namedWindow( name, WINDOW_AUTOSIZE );// Create a window for display.
    Mat weightcolored;
    applyColorMap(displayimg, weightcolored, COLORMAP_JET);
    DEBUG_STATEMENT(imshow(name, weightcolored));
    return;
}



void DisplayColouredDepth(Mat depthimg, Mat img)
{
    Mat colourMap(util::ORIG_ROWS,util::ORIG_COLS,CV_8UC1);
    float* depth_ptr= depthimg.ptr<float>(0);
    uchar* color_ptr=colourMap.ptr<uchar>(0);
    
    int y,x,j;
    for(y=0; y<util::ORIG_ROWS; y++ )
    {
        depth_ptr= depthimg.ptr<float>(y);
        color_ptr=colourMap.ptr<uchar>(y);
        j=0;
        for(x=0; x<util::ORIG_COLS;x++)
        {
            
            color_ptr[j++]=uchar(depth_ptr[x]*100.f);
            
            if(depth_ptr[x]*100.f > 255)
                color_ptr[j-1]=255;
            
        }
    }
    
    Mat colourMap_rgb;
    applyColorMap(colourMap, colourMap_rgb, COLORMAP_JET);
    
    uchar* colour_rgb=colourMap_rgb.ptr<uchar>(0);
    uchar* image_ptr=img.ptr<uchar>(0);
    uchar r,g,b;
    
    for(y=0;y<util::ORIG_ROWS;y++)
    {
        colour_rgb=colourMap_rgb.ptr<uchar>(y);
        image_ptr=img.ptr<uchar>(y);
        j=0;
        
        for(x=0;x<3*util::ORIG_COLS;x=x+3)
        {
            b=colour_rgb[x];
            g=colour_rgb[x+1];
            r=colour_rgb[x+2];
            
            
            if(b==128 & g==0 & r==0)
            {
                
                colour_rgb[x]=image_ptr[j];
                colour_rgb[x+1]=image_ptr[j];
                colour_rgb[x+2]=image_ptr[j];
                
            }
            j++;
        }
    }

        imshow("Frame Colour_depth",colourMap_rgb);
        waitKey(1000);
}


void DisplayWarpedImgPxelWise(Mat warpedimg,frame* prev_frame, String name, int framenum,bool homo)
{
    PRINTF("\nDisplaying Warped image");
    Mat displayimg=prev_frame->mask.clone();
    int nRows=prev_frame->currentRows;
    int nCols=prev_frame->currentCols;
    
    uchar* displayimg_ptr;
    float* warpedimg_ptr;
    
    int i,j;
    
    for(i=0;i<nRows;++i)
    {
        displayimg_ptr=displayimg.ptr<uchar>(i); //initialize to row 0 (single row Mat)
        warpedimg_ptr=warpedimg.ptr<float>(i);
        
        for(j=0;j<nCols;++j)
        {
            displayimg_ptr[j]=uchar((abs(warpedimg_ptr[j])));
        }
        
    }
    //namedWindow( name, WINDOW_AUTOSIZE );// Create a window for display.
    DEBUG_STATEMENT(imshow(name, displayimg));
    
    //save image
    stringstream ss;
    string str = "warped_img_";
    string type = ".jpg";
    
    ss<<"Test_images/"<<str<<(framenum-1)<<"_"<<int(homo)<<type;
    string filename = ss.str();
    ss.str("");
    
    imwrite(filename, displayimg);
    
    
    return;
    
}

void DisplayInitialResPixelWise(Mat residualimg,frame* prev_frame, String name, int framenum,bool homo)
{
    PRINTF("\nDisplaying Warped image");
    Mat displayimg=prev_frame->mask.clone();
    int nRows=prev_frame->currentRows;
    int nCols=prev_frame->currentCols;
    
    uchar* displayimg_ptr;
    float* residualimg_ptr;
    
    int i,j;
    
    for(i=0;i<nRows;++i)
    {
        displayimg_ptr=displayimg.ptr<uchar>(i); //initialize to row 0 (single row Mat)
        residualimg_ptr=residualimg.ptr<float>(i);
        
        for(j=0;j<nCols;++j)
        {
            displayimg_ptr[j]=uchar((abs(residualimg_ptr[j])));
        }
        
    }
    //namedWindow( name, WINDOW_AUTOSIZE );// Create a window for display.
    DEBUG_STATEMENT(imshow(name, displayimg));
    
    //save image
    stringstream ss;
    string str = "initial_res_";
    string type = ".jpg";
    
    //ss<<"/Users/himanshuaggarwal/Desktop/Test_images/"<<str<<(framenum-1)<<"_"<<int(homo)<<type;
    ss<<"Test_images/"<<str<<(framenum-1)<<"_"<<int(homo)<<type;
    string filename = ss.str();
    ss.str("");
    
    imwrite(filename, displayimg);
    
    return;
    
}

void DisplayIterationResPixelWise(Mat residualimg,frame* prev_frame, String name, int framenum,bool homo)
{
    PRINTF("\nDisplaying Warped image");
    Mat displayimg=prev_frame->mask.clone();
    int nRows=prev_frame->currentRows;
    int nCols=prev_frame->currentCols;
    
    uchar* displayimg_ptr;
    float* residualimg_ptr;
    
    int i,j;
    
    for(i=0;i<nRows;++i)
    {
        displayimg_ptr=displayimg.ptr<uchar>(i); //initialize to row 0 (single row Mat)
        residualimg_ptr=residualimg.ptr<float>(i);
        
        for(j=0;j<nCols;++j)
        {
            
            displayimg_ptr[j]=uchar((abs(residualimg_ptr[j])));
        }
        
    }
    //namedWindow( name, WINDOW_AUTOSIZE );// Create a window for display.
    DEBUG_STATEMENT(imshow(name, displayimg));
    
    //save image
    stringstream ss;
    string str = "final_res_";
    string type = ".jpg";
    
    ss<<"Test_images/"<<str<<(framenum-1)<<"_"<<int(homo)<<type;
    string filename = ss.str();
    ss.str("");
    
    imwrite(filename, displayimg);

    return;
    
}

void DisplayWeightsPixelWise(Mat weights,frame* prev_frame, String name, int framenum,bool homo)
{
    PRINTF("\nDisplaying Warped image");
    int nRows=weights.rows;
    int nCols=weights.cols;
    
    Mat displayimg=Mat::zeros(nRows, nCols, CV_8UC1);
    
    uchar* displayimg_ptr;
    float* weights_ptr;
    
    int i,j;
    
    for(i=0;i<nRows;++i)
    {
        displayimg_ptr=displayimg.ptr<uchar>(i); //initialize to row 0 (single row Mat)
        weights_ptr=weights.ptr<float>(i);
        
        for(j=0;j<nCols;++j)
        {

            displayimg_ptr[j]=uchar((abs(weights_ptr[j])*4000));
        }
        
    }
    //namedWindow( name, WINDOW_AUTOSIZE );// Create a window for display.
    Mat weightcolored;
    applyColorMap(displayimg, weightcolored, COLORMAP_JET);
    DEBUG_STATEMENT(imshow(name, weightcolored));
    
    //save image
    stringstream ss;
    string str = "weights_";
    string type = ".jpg";
    
    ss<<"Test_images/"<<str<<(framenum-1)<<"_"<<int(homo)<<type;
    string filename = ss.str();
    ss.str("");
    
    imwrite(filename, weightcolored);

    return;
    
}

void DisplayOriginalImgPixelWise(Mat origimg,frame* prev_frame, String name, int framenum,bool homo)
{
    PRINTF("\nDisplaying Original image");
    
    Mat displayimg=prev_frame->mask.clone();
    
    int nRows=prev_frame->currentRows;
    int nCols=prev_frame->currentCols;
    
    uchar* displayimg_ptr;
    uchar* origimg_ptr;
    
    int i,j;
    for(i=0;i<nRows;++i)
    {
        displayimg_ptr=displayimg.ptr<uchar>(i); //initialize to row 0 (single row Mat)
        origimg_ptr=origimg.ptr<uchar>(i);
        
        for(j=0;j<nCols;++j)
        {
            if(displayimg_ptr[j]==0)
                continue;
            
            else
                displayimg_ptr[j]=uchar((abs(origimg_ptr[j])));
        }
        
    }
    //namedWindow( name, WINDOW_AUTOSIZE );// Create a window for display.
    DEBUG_STATEMENT(imshow(name, displayimg));
    
    //save image
    stringstream ss;
    string str = "orig_img_";
    string type = ".jpg";
    
    ss<<"Test_images/"<<str<<(framenum-1)<<type;

    string filename = ss.str();
    ss.str("");
    
    imwrite(filename, displayimg);
    
    return;

}





