filename='posesabs.dat';%path of the text file having the poses. Needs to be converted into .dat
length1=1;
length2=11329-1;% length of the text file(no. of rows)

B=dlmread(filename,' ');

for i=1:length2
    count=B(i,1) % Frame number
    w1=B(i,3);
    w2=B(i,4);
    w3=B(i,5);
    
    p=[3 3 3]';% Dummy point
    
    v1=B(i,6);
    v2=B(i,7);
    v3=B(i,8);
    
    %%%%%%%%%%%% CALCULATIONS %%%%%%%%%%%%%
    se3=[0 -w3 w2 v1;w3 0 -w1 v2;-w2 w1 0 v3;0 0 0 0];
    secap=expm(se3);%Conversion from Lie algebra to Lie group(R,T)
    
    secapinv=inv(secap);
    
    R=secap(1:3,1:3);
    T=secapinv(1:3,4);
    T=10.*T; % Magnifying values for easy visualiu\zation
    ptrans=R*p;
    str='Frame count := ';
    str1=num2str(count);
    str2=strcat(str,{' '},str1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    figure(1);
    mTextBox=uicontrol('style','text','units','normalized','position',[0.33 .88 .3 .05]);
    set(mTextBox,'String',str2)
    subplot(1,2,1)
    quiver3(0,0,0,R(3,3),R(3,1),R(3,2))%for visualizing rotation.Observed rotation will be opposite to actual
    title('Rotation plot')
    grid on
    axis equal
    axis manual
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    xlim([0 1]);
    ylim([-1 1]);
    zlim([-1 1]);
    
    
    subplot(1,2,2)
    title('Translation plot')
    grid on
    axis equal
    axis manual
    view(3);
    scatter3(T(1),T(2),T(3));% X, Y, Z
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    hold on;
    
    %%%%%% UNCOMMENT IF YOU ARE USING MATLAB 2015(gives alternate visualization,not needed)%%%%%%%%%%%%%%%%%%
    %     figure(5);
    %     view(3);
    %     hold on;
    %     cam = plotCamera('Location',T,'Orientation',R,'Opacity',0);
    %     %scatter3(T(1),T(2),T(3));% Z, X, Y
    %     grid on
    %     axis equal
    %     axis manual
    %     xlabel('Z');
    %     ylabel('X');
    %     zlabel('Y');
    %     xlim([-15,20]);
    %     ylim([-15,20]);
    %     zlim([-5,30]);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    pause(0.00001);
    
end
