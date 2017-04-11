% Reading text file and arranging it in matrices as required by Rotation
% averaging(Govindu)
clearvars;
% clc;

keyframe_prop=8;
extramatch1=dlmread('matchframes.txt',' ');
extramatch2=dlmread('matchframes_globalopt.txt',' ');
extramatch2=extramatch2(1:size(extramatch2,1),:);
delim=[1 1 zeros(1,11)];
extramatch3=extramatch1;

% extramatch1_filter=[];
% filter_end=1;

% for i=1:length(extramatch1)
%     if extramatch1(i,1)-extramatch1(i,2)>=keyframe_prop||extramatch1(i,1)==1||extramatch1(i,1)==8
%         extramatch1_filter(filter_end,:)=extramatch1(i,:);
%         filter_end=filter_end+1;
%     end
% end

% extramatch1=extramatch1_filter;

extramatch2=extramatch2(abs(extramatch2(:,1)-extramatch2(:,2))~=8,:);
extramatch=[delim;extramatch1;extramatch2];
extramatch=sortrows(extramatch,1);
%extramatch=extramatch1;
Pose_abs=dlmread('poses_orig.txt',' ');
%  Pose_abs=Pose_abs(mod(Pose_abs(:,1),8)==0,:);
Pose_abs=[1 1 0 0 0 0 0 0 0 0;Pose_abs];

[rows,columns] = size(extramatch);

RR=zeros(3,3,rows);



disp('Calculating relative pose matrix(RR) as computed by loop closure...');

Index=unique([extramatch(:,1);extramatch(:,2)]);

%     structkeyset=zeros(length(Index),1);
%     structvalueset=zeros(length(Index),1);

for i=1:length(Index)
    structkeyset(i)=Index(i);
    structvalueset(i)={i};
end


structMapObj=containers.Map(structkeyset,structvalueset);


for i=1:rows
    m=[0 -extramatch(i,5) extramatch(i,4);extramatch(i,5) 0 -extramatch(i,3);-extramatch(i,4) extramatch(i,3) 0];
    RR(:,:,i)=expm(m);
end

I=zeros(2,size(extramatch,1));

for i=1:size(extramatch,1)
    I(1,i)=structMapObj(extramatch(i,2));
    I(2,i)=structMapObj(extramatch(i,1));
end

last_position=size(RR,1);


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
so3=zeros(depth1,8);


for i=1:depth1
    t=logm(RotMat(:,:,i));
    so3(i,:)=[Index(i) t(3,2) t(1,3) t(2,1) Pose_abs(i,[6 7 8 1])];
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

