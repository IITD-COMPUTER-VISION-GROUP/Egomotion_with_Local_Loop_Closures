start_id=1;
num_batches=3;
keyframe_prop=8;


i=1;
frame_id=start_id;
batch_count=0;

frame_index=[] %frame id, is_keyframe, is_transition_frame

while(batch_count<num_batches*2)
    frame_index(1,i)=frame_id;
    
    if i==1 || mod(i,keyframe_prop)==0
        frame_index(2,i)=1;
        frame(3,i)=0;
        
        if i~=1
             batch_count=batch_count+1;
        end
        
        if batch_count==num_batches
            frame_index(3,i)=1;
        end
        
    else
        frame_index(2,i)=0;
        frame(3,i)=0;
    end
    
    
       
    frame_id=frame_id+1;
    i=i+1;
end


Last_frame=1;
Base_frame=1;
Last_frame_pose=[0 0 0 0 0 0];
Pose_abs=[1 0 0 0 0 0 0];
concatenations_done=1;
keyframe_prop=8;
batch_size=3;
window=keyframe_prop*batch_size;


while true
    Pose_local=dlmread('poses_orig.txt',' ');
    if concatenations_done>1
        for i=1:window
            [R_1wrt2, T_1wrt2, SE3pose_1wrt2,lie]=concatenatePose(Pose_local(i,3:8),Pose_abs(Base_frame,2:end));
            Pose_abs=[Pose_abs;[Base_frame+i lie]];
        end
    end
    
    Pose_local=[1 1 0 0 0 0 0 0 0 0;Pose_local]
    Pose_local_copy=Pose_local;
    Relative_pose=dlmread('matchframes.txt',' ');
    Extra_pose=dlmread('matchframes_globalopt.txt',' ');
    
    if concatenations_done>1
        for i=1:length(Pose_local)
            
            [R_1wrt2, T_1wrt2, SE3pose_1wrt2,lie]=concatenateOriginPose(Pose_local(i,3:8),Pose_local(window,3:8));
            Pose_local_copy(i,3:8)=lie;
            
        end
    end
    
    if concatenations_done>1
        filtered_pose=perform_rotation_averaging(Relative_pose,Extra_pose,Pose_local_copy(25:end,:));
        Base_frame=Last_frame;
        Last_frame=Last_frame+window;
        
        
    else
        filtered_pose=perform_rotation_averaging(Relative_pose,Extra_pose,Pose_local_copy(2:end,:));
        Last_frame=window;
    end
    
    dlmwrite(strcat('Filtered_poses_',num2str(concatenations_done),'.txt'),filtered_pose,' ');
    concatenations_done=concatenations_done+1;
    
    Disp(strcat('Rotation averaging done for ',num2str(Last_frame)));
    
    
end

