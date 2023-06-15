%% Be sure to load the 3D scan as the variables 'framestack' (negatives clipped) and 'framestack_raw' (contains negatives)

%% Determine FWHM
figure;
imagesc(framestack(310:370,487:492,164));
caxis([0 5e3]);
axis image

figure;
frameline = mean(framestack_raw(310:370,487:492,164), 2);
x = (1:length(frameline))/pixels_per_um;
f = fit(x',frameline,'gauss1');

xfit = linspace(min(x), max(x), 200);
yfit = f.a1 * exp(-((xfit - f.b1) ./ f.c1).^2);
plot(x, frameline, '.-'); hold on
plot(xfit, yfit, 'k')
xlabel('x (um)')
set(gca, 'fontsize', 14)

sigma = f.c1 / sqrt(2);
FWHM = 2*sqrt(2*log(2)) * sigma;
plot(f.b1 - (FWHM/2 * [-1 1]), f.a1 * [1 1]/2, '.-r')
hold off
legend('image line', 'fit', 'FWHM')
title(sprintf('FWHM = %.3fum', FWHM))
 
