close all
xstep = -10*pi:pi/10:10*pi;
ystep = -10*pi:pi/10:10*pi;
[X,Y] = meshgrid(xstep,ystep);

n=8; % should be even integer
k=2;
s = (2*pi/10);

% Set up video writer
v = VideoWriter('output/plot_animation.mp4', 'MPEG-4');
v.VideoCompressionMethod
open(v);

figure;

% Initialize arrays to store polar plot data
theta = linspace(0, 2*pi, numel(xstep));
rho = zeros(size(xstep));

for p = 0:(pi/10):36*pi/4  % Iterate over different values of p
    clf; % Clear the current figure
    
    Z = zeros(length(xstep));
    m = 0;
    for i = ((0:(n-1))-((n-1)/2))
        [X1,Y1] = meshgrid(xstep+(i*pi/k),ystep);
        Z = Z + (waveCosPh(X1,Y1,p+m*s,p));
        m = m+1;
    end

    subplot(1,2,1);
    surface(X,Y,Z,'edgecolor','none');
    xlim([-10*pi 10*pi]);ylim([-10*pi 10*pi]);zlim([-10,10]);
    view([-40 70]);
    set(gca,'xticklabel',[]);
    set(gca,'yticklabel',[]);
    set(gca,'zticklabel',[]);
    set(gca,'xtick',[]);
    set(gca,'ytick',[]);
    set(gca,'ztick',[]);

    subplot(1,2,2);
    surface(X,Y,Z,'edgecolor','none');
    xlim([-10*pi 10*pi]);ylim([-10*pi 10*pi]);zlim([-10,10]);
    view([0 90]);
    set(gca,'xticklabel',[]);
    set(gca,'yticklabel',[]);
    set(gca,'zticklabel',[]);
    set(gca,'xtick',[]);
    set(gca,'ytick',[]);
    set(gca,'ztick',[]);
    
    % Add red dots on top of the 2D plot
    hold on;
    for i = ((0:(n-1))-((n-1)/2))
        plot3(i*pi/k, 0, 3, 'r.', 'MarkerSize', 10); % set the z-value to 3
    end
    hold off;

    drawnow; % Force MATLAB to refresh the plot
    
    % Write frame to video
    frame = getframe(gcf);
    writeVideo(v, frame);
    
    % Collect data for polar plot
    rho = rho + sum(Z, 2);
end

close(v); % Close the video writer

% Create polar plot
figure;
polarplot(theta, rho);

function z = waveCosPh(x,y,Ph,r)
    d = sqrt(x.^2 + y.^2) - Ph;
    d(d > r) = pi/2;
    z = cos(d);
end
