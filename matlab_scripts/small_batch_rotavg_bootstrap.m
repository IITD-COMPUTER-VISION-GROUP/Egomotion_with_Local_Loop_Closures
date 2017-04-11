%call pose abs
disp('In small batch rotavg')
clear
clc
close all

addpath('se32SE3')
addpath('SO3GraphAveraging')

is_bootstrap =1;

% Pose_abs(:,1)=Pose_abs(:,1)+1;% :\ Incrmenting g=frame id
if is_bootstrap
    disp('Bootstrapping in small batch rotavg!')
    Pose_abs=dlmread('../outputs/poses_orig.txt');%:\
    Pose_abs=Pose_abs(:,[1 3:8]);%:\
    identity=[1 0 0 0 0 0 0];%:\
    Pose_abs=[identity; Pose_abs]; %:\
    save('../outputs/World_pose.mat','Pose_abs')%:\
end


load('../outputs/World_pose.mat');

save('../outputs/World_pose_2.mat','Pose_abs');%%% BACKUP OF POSES

batch_size=10;
kf_prop=8;

if is_bootstrap
    transition_index=1;
else
    transition_index=kf_prop*(batch_size);% :\:\
end

Pose_local=dlmread('../outputs/poses_orig.txt');
base_frame=Pose_local(1,2);

Pose_local=[base_frame base_frame 0 0 0 0 0 0 0 0;Pose_local];
transition_frame_id=Pose_local(transition_index)
row_local=size(Pose_local,1);


%concatenating final poses w.r.t base frame
for i=2:transition_index
    [R_1wrt2, T_1wrt2, SE3pose_1wrt2,lie]=concatenatePose(Pose_local(i,3:8),Pose_abs(base_frame,2:end));
    
    Pose_abs(base_frame+i-1,:)=[base_frame+i-1,lie];
end

save('../outputs/World_pose.mat','Pose_abs');


%separating out poses for rot avg

%step 1-> make w.r.t transition frame
Pose_rot_avg=[];
k=1;
for i=transition_index+1:row_local
    
    [R_1wrt2, T_1wrt2, SE3pose_1wrt2,lie]=concatenateOriginPose(Pose_local(i,3:8),Pose_local(transition_index,3:8));
    
    Pose_rot_avg(k,:)=[Pose_local(i,1) Pose_local(i,2)  lie 0 0];
    
    k=k+1;
end
Pose_rot_avg=[transition_frame_id transition_frame_id 0 0 0 0 0 0 0 0;Pose_rot_avg];
%step 2-> make match poses

Relative_pose=dlmread('../outputs/matchframes.txt',' ');
Relative_pose=Relative_pose(Relative_pose(:,2)>=transition_frame_id,:);
Relative_pose=Relative_pose(Relative_pose(:,1)>=transition_frame_id,:);

s=dir('../outputs/matchframes_globalopt.txt');
Extra_pose=[];
if s.bytes~=0
Extra_pose=dlmread('../outputs/matchframes_globalopt.txt',' ');
Extra_pose=Extra_pose(Extra_pose(:,2)>=transition_frame_id,:);
Extra_pose=Extra_pose(Extra_pose(:,1)>=transition_frame_id,:);
end
%make copy
Relative_pose_copy=Relative_pose;
Extra_pose_copy=Extra_pose;

%do rotation averaging
filtered_pose=perform_rotation_averaging_transition1(Relative_pose,Extra_pose,Pose_rot_avg, transition_frame_id,is_bootstrap);

if is_bootstrap
    filtered_pose_modified=filtered_pose;
    new_base_frame=8;
    for i=new_base_frame:size(filtered_pose,1)
        
        [R_1wrt2, T_1wrt2, SE3pose_1wrt2,lie]=concatenateOriginPose(filtered_pose(i,2:7),filtered_pose(new_base_frame,2:7));
        
        filtered_pose_modified(i,:)=[i lie];
        
        
    end
    
    filtered_pose_modified=filtered_pose_modified(new_base_frame:end,:);
    dlmwrite('../outputs/so3poses7.txt',filtered_pose_modified,' ');
    makeSampleFile(new_base_frame,(batch_size+1)-new_base_frame/kf_prop,0);
    
else
    %write
    dlmwrite('../outputs/so3poses7.txt',filtered_pose,' ');
    makeSampleFile(transition_frame_id,batch_size,0);
    
end

disp('Exiting from small batch rotavg')
