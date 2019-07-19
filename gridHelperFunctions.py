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
import copy
# ---------------------------

# Some useful functions
#-----------------------
def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
#-----------------------

def define1DGrid(values, centres, noise, resolution):
    # - centres are the grid peaks
    # - values are the x-points and there are 'resolution' number of indices
    wave = np.zeros((resolution,resolution))
    for i in range(len(centres)):
        wave +=  gaussian(values, centres[i], noise) 
        
    return wave
#-----------------------

def rotateGrid(x, y, angle):
    # rotate the meshgrid values
    angle = math.radians(angle)
    newx = [None for i in range(len(x))]
    newy = [None for i in range(len(y))]
    for i in range(len(x)):
        newx[i] = x[i] * math.cos(angle) - y[i] * math.sin(angle)
        newy[i] = x[i] * math.sin(angle) + y[i] * math.cos(angle)       
    return newx, newy
#-----------------------

def plotGrid(axes, x, y, z, plotrange, gridType, plotColorbar=True, gridxspacing=None, alpha=None, phase=None, angle=None, noise=None):
    axes.contourf(x,y,z)
    axes.set_xlim(plotrange)
    axes.set_ylim(plotrange)
    plt.gca().set_aspect('equal', adjustable='box')
    if plotColorbar:
        plt.colorbar()
    
    titlestring = gridType
    if gridxspacing is not None:
        titlestring += " $\delta${:}".format(gridxspacing)
    if alpha is not None:
        titlestring += r", $\alpha${:.0f}$^\circ$".format(alpha)
    if angle is not None:
        titlestring += r", $\theta${:.0f}$^\circ$".format(angle)
    if noise is not None:
        titlestring += ", $\sigma${:}".format(noise)        
    if phase is not None:
        titlestring += r", $\varphi${:}".format(phase)
        
    axes.title.set_text(titlestring)
    # Note: pro-tip - do not call plt.show() inside a function, this stops the axes from being reusable outside of that function.
    
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

def gridSpacingWarning(gridxspacing, maxgridxspacing):
    # Ensure that we can properly tile the entire gridrange with no gaps after rotating the lattice.
    if gridxspacing >= maxgridxspacing:
        print("Error: you cannot define a grid spacing larger than {:}. Using maximum spacing value of {:} instead.".format(maxgridxspacing,maxgridxspacing))
        gridxspacing = maxgridxspacing
    return gridxspacing

#------------------------------------

def rotateWholeGrid(x,y, angle, coordMethod=1):
    # This function rotates the meshgrid coordinates x,y for plotting and grid centre generation (not the same thing) by 'angle' using one of two methods.
    if coordMethod==0:
        # METHOD 0: rotate the entire meshgrid map to visualize the rotation clearly and have a new coordinate system, x',y'  (xnew, ynew)
        xZ, yZ = x, y     # (coordinate system that z activation is calculated in)
        xnew,ynew = rotateGrid(x,y,angle)
    else:
        # METHOD 1: rotate the underlying grid basis but keep our coordinates cardinal x,y
        xZ, yZ = rotateGrid(x,y,-angle)  
        xnew,ynew = x,y
    
    return xZ, yZ, xnew, ynew

#------------------------------------

def findMaxPointCentre(z):
    # This function finds the index of z ([m x m] activation map) corresponding to the centre of the most active grid cell
    a = np.argmax(z,0)
    m = np.amax(z,0)
    xloc = np.argmax(m)
    yloc = np.argmax(z[xloc])
    return xloc, yloc

#------------------------------------

def createGrid(gridAspacing, gridBspacing, alpha, noise, angle, phase, resolution, gridrange, plotrange, coordMethod=1):
    # This function will create a lattice grid of gaussian blobs according to the gridA and gridB spacings (which are the two axes defining the grid), and alpha, the angle between them. Axis A maps onto the x-axis prior to rotation, but axis B does not necessarily map onto the y-axis (only does for alpha = 90).
    # Inputs:
    # - parameters defining the grid properties
    # - 'coordMethod' determines whether any rotation is applied to the lattice by:
    #    0:   rotating the x, y values (i.e. the whole coordinate system) OR
    #    1:   (default, more intuitive and convenient) rotating the activation function z, and keeping x, y in cardinal coordinates.
    # Outputs: the x, y, and z values for that map. (xnew, ynew are the final coordinate system)
    # Notes: xZ, yZ, is the coordinate system that the z activation is calculated in (differs depending on coordMethod)
    #        - this function extends the gridranges when defining grid centres, and imposes a maximum gridspacing to ensure we can always properly tile the entire gridrange
    #        - grid angles of  90°, 72°, 51.4° and 45°  correspond to 4-fold, 5-fold, 7-fold and 8-fold rotational symmetry
    # Define coordinates for our grid
    values = np.linspace(gridrange[0], gridrange[1], resolution)
    x, y = np.meshgrid(values, values) 
    
    # Ensure that we can properly tile the entire gridrange with no gaps after rotation.
    maxgridAspacing = 10
    gridAspacing = gridSpacingWarning(gridAspacing, maxgridAspacing)
    gridBspacing = gridSpacingWarning(gridBspacing, maxgridAspacing)
        
    # Extend the space of included x,y,z so that our map is definitely uniformly tiled after the rotation 
    extgridrange = (gridrange[0] - maxgridAspacing, gridrange[1] + maxgridAspacing)  # (being safe)
    
    # Define the grid centres for the x- and y-dimension  (we will separate odd and even rows to create a hex grid)
    gridyspacing = gridBspacing * math.sin(math.radians(alpha)) * 2                                     # y-space (cartesian, not along axis B) between odd rows 
    gridxcentres_odd = np.arange(extgridrange[0]+phase, extgridrange[1], gridAspacing)              
    gridycentres_odd = np.arange(extgridrange[0], extgridrange[1], gridyspacing)
    gridxcentres_even = np.arange(extgridrange[0]+phase+(gridAspacing*math.cos(math.radians(alpha))), extgridrange[1], gridAspacing)  
    gridycentres_even = np.arange(extgridrange[0]+gridyspacing/2, extgridrange[1], gridyspacing)

    # Rotate the grid (default method: 1)
    xZ,yZ,xnew,ynew = rotateWholeGrid(x,y,angle)
    
    # Apply the hexagonal grid activation function to the z-values
    z_odd = define1DGrid(xZ, gridxcentres_odd, noise, resolution) * define1DGrid(yZ, gridycentres_odd, noise, resolution)
    z_even = define1DGrid(xZ, gridxcentres_even, noise, resolution) * define1DGrid(yZ, gridycentres_even, noise, resolution)
    z = z_odd + z_even
    
    return xnew, ynew, z

#------------------------------------

def fitGridFunc(x, gridAspacing, gridBspacing, alpha, noise, angle, phase):
    # This function will create a lattice grid of gaussian blobs according to the gridA and gridB spacings (which are the two axes defining the grid), and alpha, the angle between them. Axis A maps onto the x-axis prior to rotation, but axis B does not necessarily map onto the y-axis (only does for alpha = 90).
    # Inputs:
    # - parameters defining the grid properties
    # - 'coordMethod' determines whether any rotation is applied to the lattice by:
    #    0:   rotating the x, y values (i.e. the whole coordinate system) OR
    #    1:   (default, more intuitive and convenient) rotating the activation function z, and keeping x, y in cardinal coordinates.
    # Outputs: the x, y, and z values for that map. (xnew, ynew are the final coordinate system)
    # Notes: xZ, yZ, is the coordinate system that the z activation is calculated in (differs depending on coordMethod)
    #        - this function extends the gridranges when defining grid centres, and imposes a maximum gridspacing to ensure we can always properly tile the entire gridrange
    #        - grid angles of  90°, 72°, 51.4° and 45°  correspond to 4-fold, 5-fold, 7-fold and 8-fold rotational symmetry
    
    gridrange = [-8,8]
    resolution = 100
    maxgridAspacing = 10
    extgridrange = (gridrange[0] - maxgridAspacing, gridrange[1] + maxgridAspacing)  # (being safe)
    
    y = copy.deepcopy(x)
    
    # Define the grid centres for the x- and y-dimension  (we will separate odd and even rows to create a hex grid)
    gridyspacing = gridBspacing * math.sin(math.radians(alpha)) * 2                                     # y-space (cartesian, not along axis B) between odd rows 
    gridxcentres_odd = np.arange(extgridrange[0]+phase, extgridrange[1], gridAspacing)              
    gridycentres_odd = np.arange(extgridrange[0], extgridrange[1], gridyspacing)
    gridxcentres_even = np.arange(extgridrange[0]+phase+(gridAspacing*math.cos(math.radians(alpha))), extgridrange[1], gridAspacing)  
    gridycentres_even = np.arange(extgridrange[0]+gridyspacing/2, extgridrange[1], gridyspacing)

    # Rotate the grid (default method: 1)
    xZ,yZ,xnew,ynew = rotateWholeGrid(x,y,angle)
    
    # Apply the hexagonal grid activation function to the z-values
    z_odd = define1DGrid(xZ, gridxcentres_odd, noise, resolution) * define1DGrid(yZ, gridycentres_odd, noise, resolution)
    z_even = define1DGrid(xZ, gridxcentres_even, noise, resolution) * define1DGrid(yZ, gridycentres_even, noise, resolution)
    z = z_odd + z_even
    
    return z.ravel()*10000

#------------------------------------

def createHexGrid(gridxspacing, noise, angle, phase, resolution, gridrange, plotrange, coordMethod=1):
    # Specify a special case (hexagonal) of a general grid lattice
    return createGrid(gridxspacing, gridxspacing, 60, noise, angle, phase, resolution, gridrange, plotrange, coordMethod)
#------------------------------------

def createSquareGrid(gridxspacing, noise, angle, phase, resolution, gridrange, plotrange, coordMethod=1):
    # Specify a special case (hexagonal) of a general grid lattice
    return createGrid(gridxspacing, gridxspacing, 90, noise, angle, phase, resolution, gridrange, plotrange, coordMethod)

#------------------------------------
def createRectGrid(gridxspacing, gridyspacing, noise, angle, phase, resolution, gridrange, plotrange, coordMethod=1):
    # Specify a special case (hexagonal) of a general grid lattice
    return createGrid(gridxspacing, gridyspacing, 90, noise, angle, phase, resolution, gridrange, plotrange, coordMethod)

#------------------------------------
