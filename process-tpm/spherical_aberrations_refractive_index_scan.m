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
E_SA_booth_sodalime = SA(n_1, n_2_sodalime, d_nom, lambda, NA, num_pixels);
n_2_range = (1.47):0.001:(1.56);
corrs = zeros(size(n_2_range));

% Scan over refractive index
for k = 1:length(n_2_range)
    E_SA_booth = SA(n_1, n_2_range(k), d_nom, lambda, NA, num_pixels);
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



function E_SA_booth = SA(n_1, n_2, d_nom, lambda, NA, num_pixels)
    x = linspace(-1, 1, num_pixels);
    y = x';
    rho = sqrt(x.^2 + y.^2);
    mask2D = (rho < 1);
    
    
    % analytical calculation of the spherical aberration correction according
    % to P. S. Salter, M. Baum, I. Alexeev, M. Schmidt, and M. J. Booth, 
    % “Exploring the depth range for three-dimensional laser machining with aberration correction,” 
    % Opt. Express, vol. 22, no. 15, pp. 17644–17656, 2014.
    
    % n_1       refractive index of medium closest to microscope objective
    % n_2       refractive index of medium of your sample
    % lambda    vacuum wavelength of light
    % d_nom     how deep you're aiming in material n_2
    % NA        NA of your objective
    % mask2D    circular mask of everything that falls within your NA
    %%
    rho = sqrt(x.^2 + y.^2);
    
    k0 = 2*pi/lambda;
    kz_glass = k0 * sqrt(n_2^2 - (NA * rho).^2);
    kz_water = k0 * sqrt(n_1^2 - (NA * rho).^2);
    
    dkz = kz_water - kz_glass; %phi_SA in paper
    phi_SA_prime = dkz - mean(dkz(mask2D));
    
    D_n2 = kz_glass;
    D_n2_prime = D_n2 - mean(D_n2(mask2D));
    
    %% calculate s
    numerator = phi_SA_prime .* D_n2_prime;
    denominator = D_n2_prime.^2;
    s_factor = mean(numerator(mask2D)) / mean(denominator(mask2D));
    s = 1/(1 + s_factor);
    
    d_act = d_nom/s;
    
    %% calculate analytical SA correction where the dot product is not dependent on d_nom
    SA_defocusfree = d_nom * kz_water - d_act * kz_glass;
    
    % This is the defocus free SA correction with mask, where we subtract the center
    SA_defocusfree_shifted = SA_defocusfree-SA_defocusfree(round(end/2),round(end/2)); % this is SA_defocusfree minus the mean of the center value.
    
    E_SA_booth = mask2D.*exp(-1i*SA_defocusfree_shifted);   % Electric field from Booth paper
end
