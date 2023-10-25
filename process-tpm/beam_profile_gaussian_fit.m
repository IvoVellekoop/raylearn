% Manually set path to beam image
% Manually set correct pixel size if necessary

% 1x Magnification from SLM to SLM-Cam
% This was verified in: /ad.utwente.nl/TNW/BMPI/Data/Daniel Cox/ExperimentalData/raylearn-data/TPM/calibration/2022-05-10_magnification-SLM-cam/
% Error was only 0.1%

pix_size_m = 5.5e-6;

% Manually load
beam_img = imread('/ad.utwente.nl/TNW/BMPI/Data/Daniel Cox/ExperimentalData/raylearn-data/TPM/calibration/intensity-beam-profile-slmcam-basler-acA2040-90umNIR.png');

fig=figure(1);
imagesc(beam_img)
title('Please select the beam center')
pos = ginput(1);    % (dim2, dim1)

x_m = pix_size_m .* (1:size(beam_img, 2))';
y_m = pix_size_m .* (1:size(beam_img, 1))';
Ix = beam_img(round(pos(2)), 1:end)';
Iy = beam_img(1:end, round(pos(1)));

fit_Ix = fit(x_m, double(Ix), 'gauss1');
fit_Iy = fit(y_m, double(Iy), 'gauss1');

%% Plot fit
figure(2)
gx = gauss(x_m, fit_Ix.a1, fit_Ix.b1, fit_Ix.c1);
gy = gauss(y_m, fit_Iy.a1, fit_Iy.b1, fit_Iy.c1);
plot(x_m*1e3, Ix, '.', 'Color', [0.4 0.4 0.4]); hold on
plot(y_m*1e3, Iy, '.', 'Color', [1.0 0.4 0.4]);
plot(x_m*1e3, gx, 'k', 'LineWidth', 4);
plot(y_m*1e3, gy, 'r', 'LineWidth', 4); hold off
xlabel('x or y (mm)')
ylabel('Intensity')

title([sprintf('Beam profile at SLM\n') ...
    '|E|^2 = I = a * exp[-((x-b) / c)^2]  \rightarrow'...
    sprintf(' c_x=%.3gmm, c_y=%.3gmm\n', fit_Ix.c1*1e3, fit_Iy.c1*1e3)...
    '|E| = a_{|E|} * exp[-((x-b) / c_{|E|})^2]  \rightarrow'...
    sprintf(' c_{|E|,x}=%.3gmm, c_{|E|,y}=%.3gmm', sqrt(2)*fit_Ix.c1*1e3, sqrt(2)*fit_Iy.c1*1e3)])

% Resize and center figure, fontsize
set(gcf, 'Position', [200 200 900 600])
movegui('center')
set(gca, 'FontSize', 14)


function g = gauss(x, a1, b1, c1)
    g = a1 * exp(-((x-b1) ./ c1).^2);
end

