function [R_1wrt2, T_1wrt2, SE3pose_1wrt2,lie]=concatenateOriginPose(se3pose_1wrt0, se3pose_2wrt0)

[R_1wrt0, T_1wrt0, SE3pose_1wrt0]=se3_2_SE3(se3pose_1wrt0);
[R_2wrt0, T_2wrt0, SE3pose_2wrt0]=se3_2_SE3(se3pose_2wrt0);

%shortcut for b*inv(A)= b/A : SE3pose_1wrt0*inv(SE3pose_2wrt0);
SE3pose_1wrt2=SE3pose_1wrt0/SE3pose_2wrt0;
lie=SE3_2_se3(SE3pose_1wrt2);
R_1wrt2=SE3pose_1wrt2(1:3,1:3);
T_1wrt2=SE3pose_1wrt2(1:3,4);    

end