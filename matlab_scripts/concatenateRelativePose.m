function [R_1wrt3, T_1wrt3, SE3pose_1wrt3]=concatenateRelativePose(se3pose_1wrt2, se3pose_2wrt3)

[R_1wrt2, T_1wrt2, SE3pose_1wrt2]=se3_2_SE3(se3pose_1wrt2);
[R_2wrt3, T_2wrt3, SE3pose_2wrt3]=se3_2_SE3(se3pose_2wrt3);

SE3pose_1wrt3=SE3pose_1wrt2*SE3pose_2wrt3;
R_1wrt3=SE3pose_1wrt3(1:3,1:3);
T_1wrt3=SE3pose_1wrt3(1:3,4);    

end