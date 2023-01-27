% Correlation spherical aberrations as a function of refractive index mismatch
close all; clear; clc


% Settings
num_pixels = 400;

n_1 = 1.3304;
n_2_sodalime = 1.5132;
d_nom = 1e-3;
lambda = 1040e-9;
NA = 0.8;

% Compute
E_SA_booth_sodalime = defocusfree_spherical_aberrations(n_1, n_2_sodalime, d_nom, lambda, NA, num_pixels);
n_2_range = (1.47):0.001:(1.56);
corrs = zeros(size(n_2_range));

% Scan over refractive index
for k = 1:length(n_2_range)
    E_SA_booth = defocusfree_spherical_aberrations(n_1, n_2_range(k), d_nom, lambda, NA, num_pixels);
    corrs(k) = corr(E_SA_booth(:), E_SA_booth_sodalime(:));
end

plot(n_2_range, abs(corrs))
hold on
xline(n_2_sodalime)
hold off
legend('correlation', 'Soda Lime Glass')
title(sprintf('Correlation spherical aberrations\nd\\_nom=%.2fmm, n_1=%.3f, NA=%.2f', ...
    1e3*d_nom, n_1, NA))
xlabel('Refractive Index')
ylabel('Correlation')
set(gca, 'fontsize', 12)
grid on

