clear; close all; clc
load /home/dani/LocalData/raylearn-data/TPM/pattern533µm.mat

num_pixels = size(field_SLM, 1);

n_1 = 1.3304;
n_2 = 1.5185;
d_nom = total_coverslip_thickness - 170e-6;
lambda = wavelength_m;
NA = 0.8;
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

%%
numerator_SLM = field_SLM .* D_n2_prime;
s_factor_SLM = mean(numerator_SLM(mask2D)) / mean(denominator(mask2D));
field_SLM_defocusfree = field_SLM - s_factor_SLM .* D_n2_prime;

%% Correlation
correlation = corr(E_SA_booth(:), field_SLM(:) .* mask2D(:));
abs(correlation)

%% Phase matched
field_SLM_phase_matched = field_SLM .* exp(-1i * angle(correlation));

%% Plot
figure;
imagesc(angle(E_SA_booth))
colorbar
title('Salter and Booth phase')

figure;
imagesc(angle(field_SLM_phase_matched .* mask2D))
colorbar
title(sprintf('Phase SLM raylearn\nOBJ1 zshift=%.2fum, corr=%.2f', obj1_zshift*1e6, abs(correlation)))

figure;
plot(unwrap(angle(field_SLM_phase_matched(end/2,:))), 'r'); hold on
plot(unwrap(angle(E_SA_booth(end/2,:))), 'k'); hold off
legend('raylearn', 'Booth')
title(sprintf('Unwrapped comparison\nOBJ1 zshift=%.2fum, corr=%.2f', obj1_zshift*1e6, abs(correlation)))
ylabel('unwrapped phase')
xlabel('x (SLM pixels)')