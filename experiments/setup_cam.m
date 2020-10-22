%% connect to camera and set settings (this code might be different for other cameras)
% connect to Fourier plane camera
disp('Setting up camera 1...')
camopt_ft.ExposureTime = 1/60*10^6;
camopt_ft.Id = 'Camera/22357092:Basler';
cam_ft = Camera(camopt_ft);
cams(1).cam = cam_ft;
cams(1).id = camopt_ft.Id;
cams(1).name = 'Fourier Plane';

% set camera region of interest
camopt_ft.Width = 1024;
camopt_ft.Height = 1024;
camopt_ft.OffsetX = 420;
camopt_ft.OffsetY = 35;
cam_ft.setROI([camopt_ft.OffsetX, camopt_ft.OffsetY, camopt_ft.Width, camopt_ft.Height]);

% connect to image plane camera
disp('Setting up camera 2...')
camopt_img.ExposureTime = 1/60*10^6;
camopt_img.Id = 'Camera/23108087:Basler';
cam_img = Camera(camopt_img);
cams(2).cam = cam_img;
cams(2).id = camopt_img.Id;
cams(2).name = 'Image Plane';

% set camera region of interest
camopt_img.Width = 512;
camopt_img.Height = 512;
camopt_img.OffsetX = (cam_img.get('WidthMax') - camopt_img.Width)/2;
camopt_img.OffsetY = (cam_img.get('HeightMax') - camopt_img.Height)/2;
cam_img.setROI([camopt_img.OffsetX, camopt_img.OffsetY, camopt_img.Width, camopt_img.Height]);

% camera axes (in um)
x1 = 5.5 * 75/150 * (0:camopt_ft.Width-1);
y1 = 5.5 * 75/150 * (0:camopt_ft.Height-1);
x2 = 4.8 * 1.65/75 * (0:camopt_img.Width-1);
y2 = 4.8 * 1.65/75 * (0:camopt_img.Height-1);
