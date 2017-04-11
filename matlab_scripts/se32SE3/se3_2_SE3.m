function [R T pose]=se3_2_SE3(Pose_abs)

current_skew=[0 -Pose_abs(3) Pose_abs(2) Pose_abs(4);...
    Pose_abs(3) 0 -Pose_abs(1) Pose_abs(5);...
    -Pose_abs(2) Pose_abs(1) 0 Pose_abs(6);...
    0 0 0 0];
current_se3=expm(current_skew);
R=current_se3(1:3,1:3);
T=current_se3(1:3,4);
pose=current_se3;
end