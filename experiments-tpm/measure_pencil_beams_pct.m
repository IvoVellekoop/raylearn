%%% script used to measure the transmission in PCT setup with SLM in
%%% Fourier plane and camera in Fourier plane

clear all; close all;
%% Load camera and SLM settings
setuppct;

%%

bggrad_patch_id = 1;                % Background Gradient Patch ID
phase_patch_id = 2;                 % optimized SLM patch ID

diameter = 0.56;                    % diameter of circular SLM geometry
slm_offset_x = 0.02;                % horizontal offset of rectangle SLM geometry
slm_offset_y = 0.05;                % vertical offset of rectangle SLM geometry
N_diameter = 8;                     % number of segments on SLM diameter
resizefactor = 0.5;                 % Factor to resize measured frame before saving

% Grating parameters
bgtype = 'blaze';                   % Type of grating ('sine', 'blaze', 'square')
ppp = 3;                            % Pixels per period
pmin = 0;                           % Minimum phase
pmax = 255;                         % Maximum phase
xSLM = 1280;
ySLM = 1024;


% experimental paramaters
p_steps = 4;
% N = 56;         % number of segments per dimension
%% Measure transmission matrix across whole camera
% initialization
showcam = 1;

refframefull = zeros(copt1.Width, copt1.Height, p_steps);
refframe = zeros(copt1.Width*resizefactor, copt1.Height*resizefactor, p_steps);

phase_set = zeros(1,1,1,p_steps);
phase_set(1,1,1,:) = round( (0:p_steps-1) * 256/p_steps);

slm.setRect(phase_patch_id, [0 0 1 1]); % Cover full SLM with phase patch
slm.setData(phase_patch_id, 0)
slm.setData(bggrad_patch_id, 0)
slm.update; slm.wait(slm.t_idle + slm.t_settle);
% Get reference frame
    
for p = 1:p_steps
    % set new value to SLM segment
%         wavefront(n) = phase_set(p);
    slm.setData(phase_patch_id, phase_set(p)); 
    slm.update; slm.wait(slm.t_idle + slm.t_settle);

    % record frame
    cam.trigger();
    refframefull(:,:,p) = double(cam.getData());
    
    if resizefactor ~= 1
        refframe(:,:,p) = imresize(refframefull(:,:,p), resizefactor);
    else
        refframe(:,:,p) = refframefull(:,:,p);
    end
    
    if showcam
        % Show what the camera sees
        figure(3);
        imagesc(refframe(:,:,p));
        axis image;
        colorbar
        title(sprintf('Reference frame | phase=%i', phase_set(p)))
        drawnow
    end
end

% set block geometry on the slm_patch and gradient on full background
bg_pix = ySLM;                      % Number of SLM pixels in gradient direction
[rects, N] = SLM.BlockedCircleSegments(N_diameter, diameter, slm_offset_x, slm_offset_y);
slm.setRect(bggrad_patch_id, [0 0 1 1]);

% Construct a grating of type bgtype
bg = bg_grating(bgtype, ppp, pmin, pmax, bg_pix);

Tset = zeros(copt1.Width*resizefactor, copt1.Height*resizefactor, N);

frames = zeros(copt1.Width*resizefactor, copt1.Height*resizefactor, N, p_steps);

slm.setData(bggrad_patch_id, bg);                   % Set background phase gradient (for getting rid of background light)

starttime = now;

% perform measurement
for n = 1:N
% for n = 35
    slm.setRect(phase_patch_id, rects(n, :))        % Set phase stepping patch to segment n
    
    for p = 1:p_steps
        % set new value to SLM segment
%         wavefront(n) = phase_set(p);
        slm.setData(phase_patch_id, phase_set(p)); 
        slm.update; slm.wait(slm.t_idle + slm.t_settle);
        
        % record frame
        cam.trigger();
        
        if resizefactor ~= 1
            frame = imresize(double(cam.getData()), resizefactor);
        else
            frame = double(cam.getData());
        end
        
        frames(:,:,n,p) = frame;
        
        if showcam
            % Show what the camera sees
            figure(3);
            imagesc(frames(:,:,n,p));
            axis image;
            colorbar
            title(sprintf('k-frame=%i/%i | phase=%i', n, N, phase_set(p)))
            drawnow
        end
    end
    % set SLM segment back to 0 phase
%     wavefront(n) = 0;
    
    % calculate field from phase stepping frames    
    Tset(:,:,n) = sum(frames(:,:,n,:).*exp(1.0i*phase_set*2*pi/255),4);
    
    eta(n, N, starttime, 'console', 'Performing phase stepping on segments', 0);
end
slm.update;

%% Save everything
dirconfigpct
save(fullfile(expdatadir, sprintf('TM-%f.mat', now)), '-v7.3')

%% 
% %% phase
% phase1 = zeros(copt1.Width,copt1.Height,N);
% M = 1;
% for k = 1:p_steps:p_steps*N
%     I1 = frames(:,:,k);
%     I2 = frames(:,:,k+1);
%     I3 = frames(:,:,k+2);
%     I4 = frames(:,:,k+3);
% %     figure(2), imagesc(I3), axis image, colormap gray, drawnow
%     phase1(:,:,M) = atan((I4 - I2) ./ (I1 - I3));
%         figure(33), imagesc(phase1(:,:,M)), axis image, colormap jet, drawnow
% 
% %     figure(2), imagesc(I4-
%     M = M + 1;
% end
%     
% 
% %% grating pattern on background
% 
% wavefront = zeros(N,1);
% phase_set = zeros(1,1,p_steps);
% phase_set(1,1,:) = round( (0:p_steps-1) * 256/p_steps);
% 
% Tset = zeros(copt1.Width,copt1.Height,N);
% grating = round(255*((cos(1:500)+1))*0.5);
% 
% mm = round(rand(5,5)*255);
% slm.setData(0,grating); slm.update;
% slm.setData(0,shift_img); slm.update;
% 
% slm_patch = 0
% % perform measurement
% for n = 56
%     
% %     frames = zeros(copt1.Width,copt1.Height,p_steps);
% 
%     for p = 4%1:p_steps
%         % set new value to SLM segment
%         wavefront(n) = phase_set(p);
%         slm.setData(slm_patch,wavefront); 
%         slm.update; 
%         
%         
%     end
% %     pause(0.5)
%         wavefront(n) = 0;
% 
%     
% end
% % MMmatrix = rand(100,100)*255;
% % figure(3), imagesc(MMmatrix)
% % slm.setData(slm_patch,wavefront);
% % slm.update
% %%
% cam2.stopPreview
%  cam2.trigger(); 
%         
% figure(3), imagesc(double(cam2.getData())), axis image, colormap gray
% cam2.startPreview
% %%
% %%
% %%
% %% 
% 
% 
% %%
% displayPort = 2;
% slm = SLM(displayPort);
% slm.setData(0,0); slm.update;
% slmxsize = 1024;
% slmysize = 1280;
% %% Temporary
% light_radius = 256;   % in pixel
% 
% [yy, xx] = meshgrid(1:slmysize, 1:slmxsize);
% megapixel_space = ones(slmxsize, slmysize);
% max_graylevel = 255;
% periodY = 2;
% kx = 2*pi/periodY;
% bg_grating = round( max_graylevel*( (cos(kx.*yy)+1)/2 ) );
% 
% pixel_width = 50;
% 
% 
% pixel_startx = 480;
% pixel_starty = 800;
% 
% offsetx = 0;  % offset from center of the SLM,  match the ROI to SLM 
% offsety = 0;
% centerSLMx = slmxsize/2+1;  % SLM usually has even number of pixels
% centerSLMy = slmysize/2+1;
% 
% macro_pixel_space = zeros(slmxsize, slmysize);
% 
% centerSLMx = 513;
% cetnerSLMy = 641;
% macro_pixel_space(centerSLMx+(-fix(pixel_width/2):round(pixel_width/2)-1),centerSLMy+(-fix(pixel_width/2):round(pixel_width/2)-1)) = 1;
% % figure(4), imagesc(macro_pixel_space), axis image
% 
% macro_pixel_space(pixel_startx, pixel_starty) = 0;
% % slmdisp = bg_grating .* megapixel_space + megapixel_space(megapixel_space == 0);
% macro_pixelX = pixel_startx : pixel_startx + pixel_width - 1;
% macro_pixelY = pixel_starty : pixel_starty + pixel_width - 1;
% 
% figure(3), imagesc(bg_grating), axis image, colormap gray
% % figure(4), plot(bg_grating(512,:))
% % for ij = 0%1:1:100
% ij = 0
%     for jj = 3
%     bg_grating(macro_pixelX, macro_pixelY) = round( (jj-1) * 256/jj);%round(255/jj);  % put the gray number from LUT for 4 step phase shifting
%     shift_img = circshift(bg_grating, [0,pixel_width*ij]);
%     figure(3), imagesc(shift_img), axis image, colormap gray, drawnow
% slm.setData(0,shift_img); slm.update;
% pause(0.5)
% 
%     end
% %%
% cam.stopPreview
% 
% cam.trigger(); 
% image1 =      double(cam.getData());
% figure(3), imagesc(image1), axis image, colormap gray
% figure(1), plot(mean(image1(505:538,:),1) / mean2(image1(505:538,600:700)))
% xlabel('pixel'), ylabel('SNB')
% % end