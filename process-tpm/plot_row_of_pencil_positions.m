
% Load pencil beam position data before plotting

center_row = 16:22;   % These indices depend on the number of SLM segments and the specific pattern!

xSLM = p.rects(:,1);
ySLM = p.rects(:,2);
plot(ySLM(center_row), cam_img_row(center_row, :), '.-');
xlabel('SLM segment position (in SLM heights) = linear with sin(angle)');
ylabel('Image camera y (cam pix)')

title(sprintf('Image camera positions (1D slices from the 4D data)\nSpherical aberrations. Defocus manually compensated.'))
