"""
Important capabilities for Fulmer et. al. 2019
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Make a class for assigning data variables
class organize_data(object):
    def __init__(self, file, column, **kwargs):
        if len(kwargs) > 0:
            brightness_cut_value = 0.45
            
            if 'mag_and_galactic_correction' not in kwargs:
                kwargs['mag_and_galactic_correction'] = 0
            if 'catalog_correction' not in kwargs:
                kwargs['catalog_correction'] = 0
                
            self.original = file[column]
            self.extinction_correction = self.original + kwargs['mag_and_galactic_correction']
            
            if kwargs['leader'] == True:
                self.brightness_cut = self.extinction_correction[self.extinction_correction \
                                                                 <= brightness_cut_value]
            else:
                self.brightness_cut = self.extinction_correction[kwargs['label'].extinction_correction \
                                                                 <= brightness_cut_value]
        
            self.data = self.brightness_cut + kwargs['catalog_correction']
        
        else:
            self.data = file[column]

# Make a class for assigning model variables 
class organize_models(object):
    def __init__(self, file, log_age):
        self.data = file[file[:,1]==log_age]
        
        self.nuv = self.data[:,24] + 1.70
        self.b = self.data[:,26]
        self.v = self.data[:,27]
        
        self.nuv_v = self.nuv - self.v
        self.b_v = self.b - self.v


# Make a class for organizing data by spectral type
class organize_spectypes(object):
    def __init__(self, indices, ra, dec, nuv_v, nuv, Tstar, logL, **kwargs):
        self.ra, self.dec, self.nuv_v, self.nuv, self.Tstar, self.logL = \
        ra[indices], dec[indices], nuv_v[indices], nuv[indices], Tstar[indices], logL[indices]

        if len(kwargs) > 0:
            self.pm_vector = kwargs['pm'][indices]

# Remove stars with abnormally high proper motions
def remove_high_proper_motion(x, y, high_ra, high_dec):
    x_length = len(x)
    within_smc = np.zeros(x_length)
    for i in range(x_length):
        if (x[i] in high_ra and y[i] in high_dec):
            within_smc[i] = False
        else:
            within_smc[i] = True
            
    within_smc = [ bool(x) for x in within_smc ]
    return within_smc

# Apply region boundaries
ra1, dec1, dec2 = 21.905, -73.215, -73.43

# Make boundary conditions
def boundary_conditions(sample):
    x = sample[1]
    y = sample[2]
        
    Region_I = (x > ra1) & (y > dec1)
    Region_II = (x > ra1) & (y < dec1) & (y > dec2)
    Region_III = (x > ra1) & (y < dec2)
    Region_IV = (x < ra1) & (y < dec1) & (y > dec2)
    
    return Region_I, Region_II, Region_III, Region_IV

# Make a class for assigning region boundaries
class assign_region_boundaries(object):
    def __init__(self, sample, bounds, **kwargs):
        self.id_number, self.ra, self.dec, self.nuv_v, self.nuv, self.b_v, self.v = sample[0][bounds], sample[1][bounds], sample[2][bounds], sample[3][bounds], sample[4][bounds], sample[5][bounds], sample[6][bounds]

        if len(sample) > 7:
            self.type, self.EB_V = sample[7][bounds], sample[8][bounds]
        
        # if band == 'NUV':
            # x = self.nuv_v
            # y = self.nuv
        main_bounds = (self.nuv >= 4.5 * self.nuv_v - 3.0) & (self.nuv <= 4.5 * self.nuv_v + 3.0) & (self.nuv >= -2.5)
        self.nuv_v_main = self.nuv_v[main_bounds]
        self.nuv_main = self.nuv[main_bounds]
            
        # elif band =='B':
        #     x = self.b_v
        #     y = self.v
        main_bounds = (self.v >= 23.0 * self.b_v - 5.0) & (self.v <= 23.0 * self.b_v + 0.2) & (self.v >= -2.5)
        self.b_v_main = self.b_v[main_bounds]
        self.v_main = self.v[main_bounds]

# Functions for main sequence fit
def find_ridge_line(x, y, n_layers):
    mag_limits = np.linspace(min(y), max(y), n_layers)
    ridge_centroidsx = np.zeros(len(mag_limits)-1)
    ridge_centroidsy = np.zeros(len(mag_limits)-1)
    
    for i in np.arange(len(mag_limits) - 1):
        
        x_layer = x[(y >= mag_limits[i]) & (y <= mag_limits[i+1])]
        y_layer = y[(y >= mag_limits[i]) & (y <= mag_limits[i+1])]
        
        centroid = (sum(x_layer) / len(x_layer), sum(y_layer) / len(y_layer))
        
        ridge_centroidsx[i] = centroid[0]
        ridge_centroidsy[i] = centroid[1]
    
    return ridge_centroidsx, ridge_centroidsy

def ridge_fit(x, y, mainx, mainy, ridge_centroidsx, ridge_centroidsy):
    slope, intercept, r_value, p_value, std_err = stats.linregress(ridge_centroidsx, ridge_centroidsy)
    # print(intercept, slope)
    
    fit = intercept + mainx*slope
    return fit, intercept, slope

# Functions for plotting
def plt_parameters(title, xlabel, ylabel, y1, y2, x1, x2, legend):
    if title != False:
        plt.title(title)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    if legend == True:
        plt.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0., prop={'size':9})
        
    plt.ylim(y1, y2)
    plt.xlim(x1, x2)

def plt_spatial(title, legend):
    plt_parameters(title, 'Right Ascension (J2000)', 'Declination (J2000)', -73.66, -73.0, 23.0, 20.9, legend)

def plt_b(title, legend):
    plt_parameters(title, '(B-V)o', 'Vo', 1.7, -6.3, -0.3, 1.0, legend)
    
def plt_nuv(title, legend):
    plt_parameters(title, '(NUV-V)o', 'NUVo', 1.3, -5.6, -2.25, 5.5, legend)
    
def plot_field(x, y, label):
    if label == True:
        plt.scatter(x, y, color='grey', label="All photometered stars", s=2, marker="o")
    elif label == False:
        plt.scatter(x, y, color='grey', s=2, marker="o")
        
def plot_region(band, x, y, color, title, label, legend):
    plt.scatter(x, y, color=color, label=label, s=10, marker="o", alpha=1.0)
    plt.title(title)
    
    if band == 'nuv':
        plt_nuv(title, legend)
    elif band == 'b':
        plt_b(title, legend)
    elif band == 'spatial':
        plt_spatial(title, legend)
