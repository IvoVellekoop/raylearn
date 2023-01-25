function [E_SA_defocusfree, phase_SA_defocusfree_shifted, kz_2] = ...
    defocusfree_spherical_aberrations(n_1, n_2, d_act, wavelength, NA, num_pixels)
    % <strong>Defocus-free Spherical Aberrations</strong>
    % Analytical calculation of the defocus-free spherical aberration correction
    %
    % <strong>Implementation of this paper:</strong>
    % P. S. Salter, M. Baum, I. Alexeev, M. Schmidt, and M. J. Booth, 
    % "Exploring the depth range for three-dimensional laser machining with aberration correction,"
    % Opt. Express, vol. 22, no. 15, pp. 17644–17656, 2014.
    %
    % <strong>Inputs:</strong>
    % n_1               Refractive index of immersion medium
    % n_2               Refractive index of medium of sample
    % wavelength        Vacuum wavelength of light
    % d_act             Actual depth of focus in material n_2 (starting from interface)
    % NA                Numerical Aperture of microscope objective
    %
    % <strong>Outputs:</strong>
    % E_SA_defocusfree  Electric field of defocus-free spherical aberrations, at the pupil plane.
    %                   The edges correspond with kx/k0 = NA and ky/k0 = NA, where (kx, ky) are the
    %                   transverse components of the angular wavevector and k0 is the vacuum
    %                   angular wavenumber = 2*pi/wavelength. The amplitude of the electric field
    %                   is 1 within the NA and 0 outside the NA.
    
    % Compute coordinates and circular mask
    x = linspace(-1, 1, num_pixels);
    y = x';
    rho = sqrt(x.^2 + y.^2);        % Normalized radius: rho=1 at NA
    mask2D = (rho < 1);             % Circular mask: 1 within NA, 0 outside NA

    % Compute k-vectors
    k0 = 2*pi / wavelength;
    kz_1 = k0 * sqrt(n_1^2 - (NA * rho).^2);
    kz_2 = k0 * sqrt(n_2^2 - (NA * rho).^2);
    
    dkz = kz_1 - kz_2;              % phi_SA in paper
    phi_SA_prime = dkz - mean(dkz(mask2D));
    
    % Compute defocus
    D_n2 = kz_2;
    D_n2_prime = D_n2 - mean(D_n2(mask2D));
    
    % Compute 
    numerator = phi_SA_prime .* D_n2_prime;
    denominator = D_n2_prime.^2;
    project_coef = mean(numerator(mask2D)) / mean(denominator(mask2D)); % Projection coefficient
    s = 1/(1 + project_coef);   % Nominal/actual depth ratio
    
    d_nom = d_act .* s;         % Nominal depth = how deep you're aiming in sample
    phase_SA_defocusfree = d_nom * kz_1 - d_act * kz_2; % Defocus-free SA phase correction
    
    % Phase shift to set center pixel pathlength equal to 0
    phase_SA_defocusfree_shifted = phase_SA_defocusfree - phase_SA_defocusfree(round(end/2), round(end/2));
    
    % Use phase to compute electric field
    E_SA_defocusfree = mask2D .* exp(-1i*phase_SA_defocusfree_shifted);
end
