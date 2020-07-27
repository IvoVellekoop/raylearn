%% connect to camera and set settings (this code might be different for other cameras)
% connect to Fourier plane camera
disp('Setting up camera 1...')
copt1.ExposureTime = 1/60*10^6;
copt1.Id = 'Camera/22357092:Basler';
cam1 = Camera(copt1);
cams(1).cam = cam1;
cams(1).id = copt1.Id;
cams(1).name = 'Fourier Plane';

% set camera region of interest
copt1.Width = 1024;
copt1.Height = 1024;
copt1.OffsetX = 420;
copt1.OffsetY = 35;
cam1.setROI([copt1.OffsetX, copt1.OffsetY, copt1.Width, copt1.Height]);

% connect to image plane camera
disp('Setting up camera 2...')
copt2.ExposureTime = 1/60*10^6;
copt2.Id = 'Camera/23108087:Basler';
cam2 = Camera(copt2);
cams(2).cam = cam2;
cams(2).id = copt2.Id;
cams(2).name = 'Image Plane';

% set camera region of interest
copt2.Width = 512;
copt2.Height = 512;
copt2.OffsetX = (cam2.get('WidthMax') - copt2.Width)/2;
copt2.OffsetY = (cam2.get('HeightMax') - copt2.Height)/2;
cam2.setROI([copt2.OffsetX, copt2.OffsetY, copt2.Width, copt2.Height]);

% camera axes (in um)
x1 = 5.5 * 75/150 * (0:copt1.Width-1);
y1 = 5.5 * 75/150 * (0:copt1.Height-1);
x2 = 4.8 * 1.65/75 * (0:copt2.Width-1);
y2 = 4.8 * 1.65/75 * (0:copt2.Height-1);
