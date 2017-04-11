# Egomotion with Local Loop Closures (ELLC) 

This code is based on the paper [Computing Egomotion with Local Loop Closures for Egocentric Videos](https://arxiv.org/pdf/1701.04743.pdf) by Suvam Patra, Himanshu Aggarwal, Himani Arora, Chetan Arora, Subhashis Banerjee. It implements a robust method for camera pose estimation using short local loop closures and rotation averaging, designed specifically for egocentric videos.

### Acknowledgements

1. **LSD-SLAM**:  J. Engel, T. Schops, and D. Cremers, “LSD-SLAM: Large-Scale Direct Monocular SLAM,” in Proceedings of the European Conference on Computer Vision (ECCV), 2014, pp. 834–849.
2. **Efficient and Robust Large-Scale Rotation Averaging**: A. Chatterjee and V. M. Govindu, “Efficient and robust large-scale rotation averaging.” in Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2013, pp. 521–528.

### Dependencies

* OpenCV (3.0.0)
* Eigen (3.2.5)
* Boost (1.59.0)
* PCL (1.8.0) (if need to visualize point clouds)
* VTK (5.10.1) - for PCL

Tested on Xcode (Version 6.4)

### Data

Dataset can be downloaded from [here](https://www.dropbox.com/sh/5iq8caqzjf1qlyx/AADy71Wg3H_0tfE4XvNYr9fSa?dl=0) or the first two videos from the [HUJI EgoSeg Dataset](http://www.vision.huji.ac.il/egoseg/videos/dataset.html). In addition you can also use your own egocentric videos but remember to provide the intrinsic parameters.

### Installation
1. Install all the dependencies.
2. Compile the source code placed in **src** folder, name the generated executable **ELLC** and place it in the **bin** folder.
3. A part of the code is executed on MATLAB. Alter the **bin\ELLC_LC.sh** script as follows:

Change all occurences of

```
/Applications/MATLAB_R2015b.app/bin/matlab -nosplash -nodisplay -r
```
To

```
<MATLAB\_PATH> -nosplash -nodisplay -r
```

### Usage

1. Place the image sequence in **data** folder.
2. Update the Instrinsic parameters and other flags as described in the next section.
3. Execute. The program has two modes:
	1. Local loop closure off : Run **ELLC** executable from inside the bin folder 
	2. Local loop closure on : Run **ELLC_LC.sh** script from inside the bin folder  

### Configurable Parameters and Flags 

Parameters/ Flags can be changed in ExternVariable.h. Some of the the important ones are summarized below: 

* Intrinsic Parameters:

	* **ORIG\_COLS**: Number of columns in the original image.
	* **ORIG\_ROWS**: Number of rows in the original image.
	* **ORIG\_FX**: Focal length in X direction.
	* **ORIG\_FY**: Focal length in Y direction.
	* **ORIG\_CX**: Principal point offset in X direction. [default: ORIG_COLS/2.0]
	* **ORIG\_CY**: Principal point offset in Y direction. [default: ORIG_ROWS/2.0]
	* **distortion_parameters**: Distortion parameters.


* Display/save images: 

	* **FLAG\_DISPLAY\_IMAGES**: Displays read images, and the residual images for each Gauss Newton iteration. [default: False]
	* **FLAG\_DISPLAY\_DEPTH_MAP**: Displays the updated depth map for every keyframe [default: true] 
	* **FLAG\_SAVE\_DEPTH\_MAP**: Saves the depth map of keyframes in /Test_images. Blue is near and red is far. [default: false] 
	* **FLAG\_SAVE\_MATCH\_IMAGES**: Saves the frames matched during local loop closures in /matches (LC only) [default: False]

* Write poses: 

	* **FLAG\_WRITE\_ORIG\_POSES**: Writes original pose of each frame w.r.t world origin i.e first frame in \outputs\poses_orig.txt [default: True]  

	* **FLAG\_WRITE\_MATCH\_POSES**: Writes relative pose of extra matches during loop closure in \outputs\matchframes.txt (LC only). [default: False]

* Multi-threading: 	
	* **FLAG\_DO\_PARALLEL\_DEPTH_ESTIMATION** [default: True]
	* **FLAG\_DO\_PARALLEL\_POSE\_ESTIMATION** [default: True]


### Outputs

Final world poses in **outputs\poses_orig.txt** are saved as **Lie Algebra** elements in the following format: 

```
CurrentFrameId KeyframeId wx wy wz vx vy vz rescalingFactor depthMapOccupancy
```



	
	
	
	
	
	
	
	
	
	
	
	
