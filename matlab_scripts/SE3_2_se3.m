function [ se3 ] = SE3_2_se3( SE3 )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

se=logm(SE3);
se3=[se(3,2) se(1,3) se(2,1) se(1:3,4)'];

end

