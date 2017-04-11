% Reading text file and arranging it in matrices as required by Rotation
% averaging(Govindu)
clearvars;
% clc;
extramatch1=dlmread('match.txt',' ');
extramatch2=dlmread('matchglobalopt.txt',' ');
extramatch2=extramatch2(1:size(extramatch2,1),:);

%extramatch2=extramatch2(abs(extramatch2(:,1)-extramatch2(:,2))~=8,:);
extramatch=[extramatch1;extramatch2];
%extramatch=extramatch1;
Pose_abs=dlmread('poses.txt',' ');

Pose_abs=[1 1 0 0 0 0 0 0 0 0;Pose_abs];

[rows,columns] = size(extramatch);

RR=zeros(3,3,rows);



disp('Calculating relative pose matrix(RR) as computed by loop closure...');

for i=1:rows
    m=[0 -extramatch(i,5) extramatch(i,4);extramatch(i,5) 0 -extramatch(i,3);-extramatch(i,4) extramatch(i,3) 0];
    RR(:,:,i)=expm(m);
end

last_position=size(RR,1);
I=transpose(extramatch(:,[2 1]));

% disp('Calculating relative pose matrix(RRa) from abs pose...');
% for j=1:rows
%     current=extramatch(j,1);
%     prev=extramatch(j,2);
%     current_so3=[0 -Pose_abs(current,5) Pose_abs(current,4);Pose_abs(current,5) 0 -Pose_abs(current,3);-Pose_abs(current,4) Pose_abs(current,3) 0];
%     current_rot=expm(current_so3);
%     prev_so3=[0 -Pose_abs(prev,5) Pose_abs(prev,4);Pose_abs(prev,5) 0 -Pose_abs(prev,3);-Pose_abs(prev,4) Pose_abs(prev,3) 0];
%     prev_rot=expm(prev_so3);
%     k=current_rot*transpose(prev_rot); %Rij = Rj*Ri'
%     RRa(:,:,j)=k;
% end
% disp('Measuring error of extra matches relative poses');
% [E,e]=CompareRotations(RR,RRa);

[rows2,columns2] = size(Pose_abs);

RRt=zeros(3,3,rows2);

for i=1:rows2
    m=[0 -Pose_abs(i,5) Pose_abs(i,4);Pose_abs(i,5) 0 -Pose_abs(i,3);-Pose_abs(i,4) Pose_abs(i,3) 0];
    RRt(:,:,i)=expm(m);
end


% for i=1:rows2
%     I(1,rows+ i) =1;
%     I(2,rows+ i) = Pose_abs(i,1);
%     RR(:,:,rows+ i)= RRt(:,:,i);
% end

RotMat=AverageSO3Graph(RR,I);

[row1,col1,depth1]=size(RotMat);
so3=zeros(depth1,7);


for i=1:depth1
    t=logm(RotMat(:,:,i));
    so3(i,:)=[i t(3,2) t(1,3) t(2,1) Pose_abs(i,[6 7 8])];
end

[E,e]=CompareRotations(RotMat,RRt);

dlmwrite('so3poses7.txt',so3,' ');

u=1:length(Pose_abs);
figure(7)
hold on
plot(u,so3(:,2),'r')
plot(u,Pose_abs(:,3),'b')
figure(8)
hold off
plot(u,so3(:,3),'r')
hold on
plot(u,Pose_abs(:,4),'b')
figure(9)
hold off
plot(u,so3(:,4),'r')
hold on
plot(u,Pose_abs(:,5),'b');

% new_origin_frameid=32;
%  
%  
%  for i=new_origin_frameid:length(so3)
%      [R_1wrt2, T_1wrt2, SE3pose_1wrt2,lie]=concatenateOriginPose(so3(i,2:7), so3(new_origin_frameid,2:7));
%      so3(i,2:7)=lie;
%  end
%  
dlmwrite('so3_31_new.txt',so3(new_origin_frameid:end,:),' ');
 
 
 