extramatch=dlmread('extra_matches.dat',' ');
Pose_abs=dlmread('poses_orig.dat',' ');
Pose_abs=[1 1 0 0 0 0 0 0 0 0;Pose_abs];

[rows,columns] = size(extramatch);

RR=zeros(3,3,rows);
RRa=zeros(3,3,rows);

disp('Calculating relative pose matrix(RR) as computed by loop closure...');
for i=1:rows
    m=[0 -extramatch(i,5) extramatch(i,4);extramatch(i,5) 0 -extramatch(i,3);-extramatch(i,4) extramatch(i,3) 0];
    RR(:,:,i)=expm(m);
end

I=transpose(extramatch(:,[2 1]));

disp('Calculating relative pose matrix(RRa) from abs pose...');
for j=1:rows
    current=extramatch(j,1);
    prev=extramatch(j,2);
    current_so3=[0 -Pose_abs(current,5) Pose_abs(current,4);Pose_abs(current,5) 0 -Pose_abs(current,3);-Pose_abs(current,4) Pose_abs(current,3) 0];
    current_rot=expm(current_so3);
    prev_so3=[0 -Pose_abs(prev,5) Pose_abs(prev,4);Pose_abs(prev,5) 0 -Pose_abs(prev,3);-Pose_abs(prev,4) Pose_abs(prev,3) 0];
    prev_rot=expm(prev_so3);
    k=transpose(prev_rot)*current_rot;
    RRa(:,:,j)=k;
end
