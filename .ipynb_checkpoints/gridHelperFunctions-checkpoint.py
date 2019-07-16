# This is a series of helper functions for create simulations of grid lattices (like grid cells)
# Author: Hannah Sheahan
# Date: 16/07/2019
# Notes: 
#       - sinudoidally-defined grids have warped noise patterns if you define them over odd and even rows (dont do it!), so these simulations use gaussian functions instead, and they are more flexible anyway.
#       - [Python 3]
# Issues: N/A

# ---------------------------
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import math
import sys
import time
import random
# ---------------------------

# Some useful functions
#-----------------------
def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
#-----------------------

def define1DGrid(values, centres, noise, resolution):
    # - centres are the grid peaks
    # - values are the x-points and there are 'resolution' number of indices
    wave = [0 for i in range(resolution)]
    for i in range(len(centres)):
        wave +=  gaussian(values, centres[i], noise) 
    return wave
#-----------------------

def rotateGrid(x, y, angle):
    # rotate the meshgrid values
    newx = [None for i in range(len(x))]
    newy = [None for i in range(len(y))]
    for i in range(len(x)):
        newx[i] = x[i] * math.cos(angle) - y[i] * math.sin(angle)
        newy[i] = x[i] * math.sin(angle) + y[i] * math.cos(angle)       
    return newx, newy
#-----------------------

def plotGrid(x, y, z, plotrange, plotColorbar=True, gridxspacing=None, phase=None, angle=None, noise=None):
    h = plt.contourf(x,y,z)
    plt.xlim(plotrange)
    plt.ylim(plotrange)
    plt.gca().set_aspect('equal', adjustable='box')
    if plotColorbar:
        plt.colorbar()
    
    titlestring = "Hex:"
    if gridxspacing is not None:
        titlestring += " spacing {:}".format(gridxspacing)
    if angle is not None:
        titlestring += ", orient {:.1f}$^\circ$".format(math.degrees(angle))
    if noise is not None:
        titlestring += ", noise {:}".format(noise)        
    if phase is not None:
        titlestring += ", phase {:}".format(phase)
        
    plt.title(titlestring)
    plt.show()
#-----------------------

# Sinusoids don't work with odd/even method:
def sinusoidal(values, spacing, phase):
    # NOTE: this is not used (and shouldnt be, because defining sinusoids over odd and even rows stretches grid in y-axis)
    f = 1/spacing
    return 0.5*np.sin(2 * np.pi * f * values + phase)+0.5

    # An illustration of how defining sinusoids over even and odd rows will stretch your grid in the y-axis
    #z_sin_odd = sinusoidal(x, gridxspacing, 0) * sinusoidal(y, gridyspacing, 0)
    #z_sin_even = sinusoidal(x, gridxspacing, 0+gridxspacing/2) * sinusoidal(y, gridyspacing, gridyspacing/2)
    #z_sin = z_sin_odd + z_sin_even
    #plotGrid(xnew, ynew, z_sin, plotrange, False, gridxspacing, phase, angle, noise)

#-----------------------
def createHexGrid(gridxspacing, noise, angle, phase, resolution, gridrange, plotrange, coordMethod=1, plotFLAG=True):
    # This function will create a hexagonal lattice grid of gaussian blobs.
    # Inputs:
    # - parameters defining the grid properties
    # - 'coordMethod' determines whether any rotation is applied to the lattice by:
    #    0:   rotating the x, y values (i.e. the whole coordinate system) OR
    #    1:   (default, more intuitive and convenient) rotating the activation function z, and keeping x, y in cardinal coordinates.
    # Outputs: the x, y, and z values for that map. (xnew, ynew are the final coordinate system)
    # Notes: xZ, yZ, is the coordinate system that the z activation is calculated in (differs depending on coordMethod)
    #        - this function extends the gridranges when defining grid centres, and imposes a maximum gridspacing to ensure we can always properly tile the entire gridrange
    
    # Ensure that we can properly tile the entire gridrange with no gaps after rotation.
    maxgridxspacing = 10
    if gridxspacing >= maxgridxspacing:
        print("Error: you cannot define a grid spacing larger than {:}. Using maximum spacing value of {:} instead.".format(maxgridxspacing,maxgridxspacing))
        gridxspacing = maxgridxspacing
        
    # Extend the space of included x,y,z so that our map is definitely uniformly tiled after the rotation 
    extgridrange = (gridrange[0] - maxgridxspacing, gridrange[1] + maxgridxspacing)  # (being safe)
    
    # Rotate the meshgrid x and y values to define the grid orientation (a new x',y' coordinate system)
    values = np.linspace(gridrange[0], gridrange[1], resolution)
    x, y = np.meshgrid(values, values) 

    # Define the grid centres for the x- and y-dimension  (we will separate odd and even rows to create a hex grid)
    gridyspacing = gridxspacing * math.sin(math.radians(60)) * 2                                     # y-space between odd rows 
    gridxcentres_odd = np.arange(extgridrange[0]+phase, extgridrange[1], gridxspacing)              
    gridycentres_odd = np.arange(extgridrange[0], extgridrange[1], gridyspacing)
    gridxcentres_even = np.arange(extgridrange[0]+phase+gridxspacing/2, extgridrange[1], gridxspacing)  
    gridycentres_even = np.arange(extgridrange[0]+gridyspacing/2, extgridrange[1], gridyspacing)

    # Rotate the grid using one of two methods (default: 1)
    if coordMethod==0:
        # METHOD 0: rotate the entire meshgrid map to visualize the rotation clearly and have a new coordinate system, x',y'  (xnew, ynew)
        xZ, yZ = x, y     # (coordinate system that z activation is calculated in)
        xnew,ynew = rotateGrid(x,y,angle)

    else:
        # METHOD 1: rotate the underlying grid basis but keep our coordinates cardinal x,y
        xZ, yZ = rotateGrid(x,y,-angle)  
        xnew,ynew = x,y
    
    # apply the hexagonal grid activation function to the z-values
    z_odd = define1DGrid(xZ, gridxcentres_odd, noise, resolution) * define1DGrid(yZ, gridycentres_odd, noise, resolution)
    z_even = define1DGrid(xZ, gridxcentres_even, noise, resolution) * define1DGrid(yZ, gridycentres_even, noise, resolution)
    z = z_odd + z_even
    
    # Plot the grid
    if plotFLAG:
        plotGrid(xnew, ynew, z, plotrange, False, gridxspacing, phase, angle, noise)

    return xnew, ynew, z

#------------------------------------