function [ Filtered_pose ] = perform_rotation_averaging(extramatch1,extramatch2,Pose_abs )
% Reading text file and arranging it in matrices as required by Rotation
% averaging(Govindu)
clearvars;
% clc;

keyframe_prop=8;

delim=[1 1 zeros(1,11)];

extramatch3=extramatch1;

extramatch1_filter=[];
filter_end=1;
%removing non kf
for i=1:length(extramatch1)
    if extramatch1(i,1)-extramatch1(i,2)>=keyframe_prop||extramatch1(i,1)==1||extramatch1(i,1)==8
        extramatch1_filter(filter_end,:)=extramatch1(i,:);
        filter_end=filter_end+1;
    end
end
extramatch1=extramatch1_filter;



extramatch=[delim;extramatch1;extramatch2];
extramatch=sortrows(extramatch,1);


Pose_abs_copy=Pose_abs;
Pose_abs_copy_2=Pose_abs;

Pose_abs=Pose_abs(mod(Pose_abs(:,1),8)==0,:);
Pose_abs=[1 1 0 0 0 0 0 0 0 0;Pose_abs];


[rows,columns] = size(extramatch);
RR=zeros(3,3,rows);

disp('Calculating relative pose matrix(RR) as computed by loop closure...');

Index=unique([extramatch(:,1);extramatch(:,2)]);
for i=1:length(Index)
    structkeyset(i)=Index(i);
    structvalueset(i)={i};
end

structMapObj=containers.Map(structkeyset,structvalueset);

for i=1:rows
   [R,t,pose]=se3_2_SE3(extramatch(i,3:8));
    RR(:,:,i)=R;
end

I=zeros(2,size(extramatch,1));
for i=1:size(extramatch,1)
    I(1,i)=structMapObj(extramatch(i,2));
    I(2,i)=structMapObj(extramatch(i,1));
end

[rows2,columns2] = size(Pose_abs); %filtered only kf
RRt=zeros(3,3,rows2);
for i=1:rows2
    [R,t,pose]=se3_2_SE3(Pose_abs(i,3:8));
    RRt(:,:,i)=R;
end

 RotMat=AverageSO3Graph(RR,I);

[row1,col1,depth1]=size(RotMat);
so3=zeros(depth1,7);

for i=1:depth1
    t=logm(RotMat(:,:,i));
    so3(i,:)=[Index(i) t(3,2) t(1,3) t(2,1) Pose_abs(i,[6 7 8])]; %kf, after rot avg, wrt world
end

[E,e]=CompareRotations(RotMat,RRt);

dlmwrite('so3poses7.txt',so3,' '); %i lie pose

%display
% u=1:length(Pose_abs);
% figure(7)
% hold on
% plot(u,so3(:,2),'r')
% plot(u,Pose_abs(:,3),'b')
% figure(8)
% hold off
% plot(u,so3(:,3),'r')
% hold on
% plot(u,Pose_abs(:,4),'b')
% figure(9)
% hold off
% plot(u,so3(:,4),'r')
% hold on
% plot(u,Pose_abs(:,5),'b');


%now adding non-kf and concatenating pose

kf_error=[Index Pose_abs(:,3:8)-so3(:,2:7)];
j=0;
count=0;

%now starts from 1 
Pose_abs_copy=[1 1 0 0 0 0 0 0 0 0;Pose_abs_copy];
Pose_abs_copy_2=[1 1 0 0 0 0 0 0 0 0;Pose_abs_copy_2];
extramatch3=[delim;extramatch3];

 for i=1:length(Pose_abs_copy)
     if i==1|| mod(i,keyframe_prop)==0
        j=j+1; 
        Pose_abs_copy(i,3:8)=so3(j,2:7); %kf 
     else
        [R_2wrt0, T_2wrt0, SE3pose_2wrt0,lie]=concatenatePose(extramatch3(i,3:8),so3(j,2:7)); %non kf
         Pose_abs_copy(i,3:8)=lie(1:6); %updated rotations+trans
     end
     count=count+1;
     if count==keyframe_prop
         count=0;
     end
     
 end
 
 %shifting origin to frame number 
 new_origin_frameid=1;
 
 g=Pose_abs_copy_2-Pose_abs_copy; %error 
 
 for i=new_origin_frameid:length(Pose_abs_copy)
     [R_1wrt2, T_1wrt2, SE3pose_1wrt2,lie]=concatenateOriginPose(Pose_abs_copy(i,3:8), Pose_abs_copy(new_origin_frameid,3:8));
     Pose_abs_copy(i,3:8)=lie;
 end
 
Filtered_pose=Pose_abs_copy(new_origin_frameid:end,[1 3:8])
 
 
 




end

