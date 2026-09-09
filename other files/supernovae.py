import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
from numpy.polynomial import Polynomial as poly

def galaxies(file):
    '''
    Plotting the spectra for the different galaxies containing and marking each spectral line
    '''
    ## Loading Data
    wl, flux = np.loadtxt(file, unpack= True)

    ### Plotting
    plt.figure(figsize=(12, 6), dpi = 200) # Set quality and size of graph
    plt.plot(wl, flux, color = 'black') # Plot the graph
    plt.title('Type 1a Supernovae Spectrum'  + os.path.basename(file)) # Label axes and graph
    plt.xlabel('Observed Wavelength (Å)')
    plt.ylabel('Flux (erg/cm^2/Å)')
    # Plot the different galaxy lines
    plt.axvline(6563, linestyle='--', label='6563 Å (Hα)')
    plt.axvline(4861, color = 'black', linestyle='--', label='4861 Å (H)')
    plt.axvline(4341, color='r', linestyle='--', label='4341 Å (H)')
    plt.axvline(6548, color='pink', linestyle='--', label='6548 Å (NII)')
    plt.axvline(6583, color='orange', linestyle='--', label='6583 Å (NII)')
    plt.axvline(3727, color='purple', linestyle='--', label='3727 Å (OII)')
    plt.axvline(4959, color='aqua', linestyle='--', label='4959 Å (OIII)')
    plt.axvline(5007, color='brown', linestyle='--', label='5007 Å (OIII)')
    plt.axvline(5890, color='gold', linestyle='--', label='5890 Å (NaI)')
    plt.axvline(5896, color='olive', linestyle='--', label='5896 Å (NaI)')
    plt.axvline(6717, color='lime', linestyle='--', label='6717 Å (SII)')
    plt.axvline(6731, color='orchid', linestyle='--', label='6731 Å (NaI)')
    plt.axvline(3969, color='maroon', linestyle='--', label='3969 Å (CaII)')
    plt.axvline(3934, color='khaki', linestyle='--', label='3934 Å (CaII)')
    plt.legend()
    plt.show()


def analyze_spectrum(file_path, wl_min, wl_max, rest_line, exclude_width=25, cont_deg=1, sigma_guess=15):
    
    """
    This function finds the spectral line for each galaxy within the wavelength range we are searching for, finds its continuum and then noramlises the flux data. 
    This is then fit to a Gaussian curve to find the best fit.
    """

    # Load the file and get the data
    wl, flux = np.loadtxt(file_path, unpack=True)

    # Mask the data for the specific wavelength range
    mask = (wl >= wl_min) & (wl <= wl_max)
    wl_fit = wl[mask]
    flux_fit = flux[mask]

    # Fit the continuum
    cont_mask = (wl_fit < rest_line - exclude_width) | (wl_fit > rest_line + exclude_width)

    p = np.polyfit(wl_fit[cont_mask], flux_fit[cont_mask], deg=cont_deg)
    continuum = np.polyval(p, wl_fit)

    # Normalising the Spectrum
    flux_norm = flux_fit / continuum

    # Making the gaussian model
    def gaussian(x, A, x0, sigma, m, b):
        return 1.0 + A*np.exp(-(x-x0)**2/(2*sigma**2)) + (m*x + b)

    # Initial guesses based on sliced data
    idx_peak = np.argmax(flux_norm)
    idx_dip  = np.argmin(flux_norm)

    if (flux_norm[idx_peak] - 1) >= (1 - flux_norm[idx_dip]):
        A_guess = flux_norm[idx_peak] - 1 # A guess
        x0_guess = wl_fit[idx_peak] # x_0 guess
    else:
        A_guess = flux_norm[idx_dip] - 1
        x0_guess = wl_fit[idx_dip]

    init = [A_guess, x0_guess, sigma_guess, 0.0, 0.0] # initial guess

    # Performing the fit using Scipy's curve_fit
    popt, pcov = curve_fit(gaussian, wl_fit, flux_norm, p0=init, maxfev=20000)

    A, x0, sigma, m, b = popt
    x0_err = np.sqrt(pcov[1,1])
    z_list = []
    z_err_list = []
    # Computing the doppler shift - redshift
    z = (x0 - rest_line) / rest_line

    # Uncertainty using your required formula
    z_err = z * (np.sqrt((2) * (x0_err / x0)**2))

    print(f"x0 = {x0:.3f} ± {x0_err:.3f} Å")
    print(f"z  = {z:.6f} ± {z_err}")
    z_list.append(z)
    z_err_list.append(z_err)

    # Printing fitted parameters
    print(f"Fitted parameters: A={popt[0]}, x0={popt[1]} Å, s={popt[2]} Å, m={popt[3]} (slope) ,c={popt[4]} (offset)")

    # Uncertainty in the peak
    sigma_x0 = np.sqrt(pcov[1,1])
    print(f"Uncertainty in x0 (line center) = {sigma_x0:.2f} Å")


    # Plotting the spectrum and gaussian fit
    plt.figure(figsize=(12,4), dpi=200)
    plt.plot(wl_fit, flux_fit, 'k-', label="Data (Sliced)")
    #plt.plot(wl_fit, gaussian(wl_fit, *popt), 'r', label="Gaussian fit")
    plt.xlabel("Observed Wavelength (Å)")
    plt.ylabel("Flux")
    plt.title("Spectral Line" + os.path.basename(file_path))
    plt.grid(linestyle='--', alpha=0.4)
    plt.legend()
    plt.show()

    # Plotting the normalised and gaussian fit
    plt.figure(figsize=(12,4), dpi=200)
    plt.scatter(wl_fit, flux_norm, s=10, c='black', label="Normalized data")
    plt.plot(wl_fit, gaussian(wl_fit, *popt), 'r', label=( f"Gaussian fit\n" f"$x_0 = {x0:.2f} \\pm {x0_err:.2f}$ Å\n" f"$z = {z:.6f}$"))
    plt.axhline(1.0, ls='--', alpha=0.5)
    plt.xlabel("Observed Wavelength (Å)")
    plt.ylabel("Normalized Flux")
    plt.title("Gaussian Fit (Continuum Normalized) " + os.path.basename(file_path))
    plt.grid(linestyle='--', alpha=0.4)
    plt.legend()
    plt.show()

    return popt, pcov, z, z_err # return the uncertainty values from the coveraince matrix

mB_peak_list = []
b_mag_err = []
bmag = []

# Function for Light curves
def lc(file, z):
    """
    standardising using LCs
    """
    time, b_mag, b_mag_err = np.loadtxt(file, unpack=True, usecols=(0, 3, 4))
    # time dilation correction (rest frame)
    time_dilation = time / (1 + z) # time dilation correction (rest frame)

    # Remove 99 (invalid data)
    clean = b_mag < 99
    time = time[clean]
    time_dilation = time_dilation[clean]
    b_mag = b_mag[clean]
    b_mag_err = b_mag_err[clean]

    # Finding the approximate peak from data (min mag)
    idx0 = np.argmin(b_mag)
    t0 = time_dilation[idx0]  # use time for zoom window

    # Masking data for specific range around the peak
    mask = (time_dilation >= t0 - 20) & (time_dilation <= t0 + 40) 

    time = time[mask]
    time_dilation = time_dilation[mask]
    b_mag = b_mag[mask]
    b_mag_err = b_mag_err[mask]

    # Polynomial fit
    p = poly.fit(time_dilation, b_mag, deg=3)

    t_fine = np.linspace(time_dilation.min(), time_dilation.max(), 2000)
    b_fine = p(t_fine)

    # Plot (one plot only)
    plt.figure(figsize=(10, 6))
    plt.errorbar(time_dilation, b_mag, yerr=b_mag_err, fmt='.', color='green', label='Data (zoom)')
    plt.plot(t_fine, b_fine, color='r', label='Polyfit deg 3')
    plt.gca().invert_yaxis()
    plt.xlabel('Corrected Time (Rest-frame Julian Date)')
    plt.ylabel('B mag')
    plt.grid(linestyle='--', alpha=0.8)
    plt.title('Type 1a Supernova Light Curve ' + os.path.basename(file))
    plt.legend()
    plt.show()

    # Peak from fit
    idx_min = np.argmin(b_fine)
    t_peak = t_fine[idx_min]
    b_peak = b_fine[idx_min]

    # delta m15
    t_15 = t_peak + 15
    d_m = np.nan

    if t_15 <= time_dilation.max():
        b_15 = p(t_15)
        d_m = b_15 - b_peak
    else:
        print("Delta m_15 cannot be computed (t_peak + 15 outside data range)")
    # Return values
    return b_peak, d_m, b_mag_err
