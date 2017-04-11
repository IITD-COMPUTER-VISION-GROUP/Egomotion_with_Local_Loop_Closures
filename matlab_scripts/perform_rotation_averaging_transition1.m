function [ Filtered_pose ] = perform_rotation_averaging_transition(extramatch1,extramatch2,Pose_abs,transition_frame_id,is_bootstrap)

% Extramatch1 = [ frame_id keyframe_id Pose_wrt_kf(3:8) scale seeds +3 ]=N X 13 
% Extramatch2 = [ current_frame_id prev_keyframe_id Pose_wrt_kf(3:8) scale seeds +3 ]=N X 13 
% Pose_abs = [current_frame_id transition_frame_id Pose_wrt_tf(3:8) +2]= NX 10
% Extramatch1 (1,:)= tf+1 tf
% POse_abs tf tf
% extramatch3 tf tf
%extramatch tf tf;extramatch1;extramatch2
%so3 -> (ascending order of kf->tf)   frame_id, rotation_averaged_pose(2:7) %only kf



keyframe_prop=8;

delim=[transition_frame_id transition_frame_id zeros(1,11)]; %transition_frame_id is origin

extramatch3=extramatch1;

extramatch1_filter=[];
filter_end=1;
%removing non kf
for i=1:length(extramatch1)
    if extramatch1(i,1)-extramatch1(i,2)>=keyframe_prop %does not take 1 as kf
        extramatch1_filter(filter_end,:)=extramatch1(i,:);
        filter_end=filter_end+1;
    end
end
extramatch1=extramatch1_filter; %only kf


extramatch=[delim;extramatch1;extramatch2];
extramatch=sortrows(extramatch,1); %sorted 

Pose_abs_copy=Pose_abs;
Pose_abs_copy_2=Pose_abs;

if is_bootstrap
% selecting out kf
 Pose_abs=[Pose_abs(Pose_abs(:,1)==1,:);Pose_abs(mod(Pose_abs(:,1),8)==0,:)]; %updated condition
%Pose_abs tf tf 
else
 % %selecting out kf CHANGGGGGEEEEEDDDDDD
  Pose_abs=Pose_abs(mod(Pose_abs(:,1)-transition_frame_id,8)==0,:); %updated condition
 %Pose_abs tf tf 
end

[rows,columns] = size(extramatch);
RR=zeros(3,3,rows);

disp('Calculating relative pose matrix(RR) as computed by loop closure...');

Index=unique([extramatch(:,1);extramatch(:,2)])
for i=1:length(Index)
    structkeyset(i)=Index(i);
    structvalueset(i)={i};
end

structkeyset
structvalueset

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

RotMat=AverageSO3Graph(RR,I);
[row1,col1,depth1]=size(RotMat)
so3=zeros(depth1,7); % frame_id, rotation_averaged_pose(2:7) %only kf

for i=1:depth1
    t=logm(RotMat(:,:,i));
    so3(i,:)=[Index(i) t(3,2) t(1,3) t(2,1) Pose_abs(i,[6 7 8])]; %kf, after rot avg, wrt world
end
j=0;
extramatch3=[delim;extramatch3];%copy of extramatch1

%dlmwrite('so3poses7.txt',so3,' '); %i lie pose

for i=1:length(Pose_abs_copy) %for i=1->tf

     if mod((Pose_abs_copy(i,1)-transition_frame_id),keyframe_prop)==0 %updated condition
        j=j+1;
        Pose_abs_copy(i,3:8)=so3(j,2:7); %kf 
     else
        [R_2wrt0, T_2wrt0, SE3pose_2wrt0,lie]=concatenatePose(extramatch3(i,3:8),so3(j,2:7)); %non kf
         Pose_abs_copy(i,3:8)=lie(1:6); %updated rotations+trans
     end
     
end
 

so3

Filtered_pose=Pose_abs_copy(:,[1,3:8]); %for initialization, all poses w.r.t tf
% frame_id, pose(2:7)
%starts with tf...


end
