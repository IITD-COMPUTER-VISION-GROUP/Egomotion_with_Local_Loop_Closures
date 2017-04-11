function [R_2wrt0, T_2wrt0, SE3pose_2wrt0,lie]=concatenatePose(se3pose_2wrt1,se3pose_1wrt0)

[R_1wrt0, T_1wrt0, SE3pose_1wrt0]=se3_2_SE3(se3pose_1wrt0);
[R_2wrt1, T_2wrt1, SE3pose_2wrt1]=se3_2_SE3(se3pose_2wrt1);

SE3pose_2wrt0=SE3pose_2wrt1*SE3pose_1wrt0;
lie=SE3_2_se3(SE3pose_2wrt0);
R_2wrt0=SE3pose_2wrt0(1:3,1:3);
T_2wrt0=SE3pose_2wrt0(1:3,4);    

end