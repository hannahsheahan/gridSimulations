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
from scipy import optimize
import math
import sys
import time
import random
import copy
# ---------------------------

# Some useful functions


#-----------------------
def nanArray(size):
    # create a numpy array of size 'size' (tuple) filled with nans
    tmp = np.zeros(size)
    tmp.fill(np.nan)
    return tmp

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
    axes.set_aspect('equal', adjustable='box')
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
    axes.axis('off')
    # Note: pro-tip - do not call plt.show() inside a function, this stops the axes from being reusable outside of that function.
    
#-----------------------

# Sinusoids don't work with odd/even method:
def sinusoidal(x, A, spacing, phase, yoffset):
    # NOTE: this is not used (and shouldnt be, because defining sinusoids over odd and even rows stretches grid in y-axis)
    f = 1/spacing
    return A*np.sin(2*np.pi*f*x + phase)+yoffset

# An illustration of how defining sinusoids over even and odd rows will stretch your grid in the y-axis
#z_sin_odd = sinusoidal(x, 0.5, gridxspacing, 0, 0.5) * sinusoidal(y, 0.5, gridyspacing, 0, 0.5)
#z_sin_even = sinusoidal(x, 0.5, gridxspacing, 0+gridxspacing/2, 0.5) * sinusoidal(y, 0.5, gridyspacing, gridyspacing/2, 0.5)
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

def fitLattice(numiter, data, fitwhichparams, initparams, lowerbound=None, upperbound=None):
    # This function (in theory) fits any 2D lattice to data.
    # In practice, this is a very difficult numerical optimisation, it takes ages and often returns local 
    # minima solutions even when searching over few parameters. 
    # Should ONLY be used when search for one or two parameters max, and with high numiter to avoid local minima.
    # Lattice parameters: gridAspacing, gridBspacing, alpha, noise, angle, phase
    
    start = time.time()
    num_params = len(initparams)
    fitted_params = nanArray((numiter,num_params))
    fitted_SSE = nanArray((numiter,1))

    # Define the default bounds. *Note that this way of defining defaults (vs in function def) is critical because defaults are defined only once in python, and if they are edited later in this script (they are) then this will affect their values in the next running of this script.
    if lowerbound is None:
        lowerbound = [1, 1, 0, 0, 0, 0]
    if upperbound is None:
         upperbound=[5, 5, 120, 5, 90, 5]
    
    # Format grid data for fitting and fit a grid function to it
    resolution = data.shape[0]
    gridrange = [-8,8]
    values = np.linspace(gridrange[0], gridrange[1], resolution)
    x,y = np.meshgrid(values, values) 
    zdata = data.ravel()*10000

    # Constrain the bound on the parameters we are not fitting
    for p in range(len(fitwhichparams)):
        if fitwhichparams[p]==0:
            lowerbound[p] = initparams[p]
            upperbound[p] = initparams[p] + 0.000000001

    # Run the parameter optimisation several times from different initial conditions
    print("Running optimisation...")        
    for i in range(numiter):
        
        try:
            params, params_covariance = optimize.curve_fit(fitGridFunc, x, zdata, p0=initparams, bounds=(lowerbound,upperbound))
            SSE = sum((fitGridFunc(x, *params) - zdata)**2)
            fitted_params[i] = params
            fitted_SSE[i] = SSE
        except RuntimeError:
            print("\n Error: max number of fit iterations hit. Moving on...")

        # display our optimisation iteration progress
        j = (i + 1) / numiter
        sys.stdout.write('\r')
        sys.stdout.write("[%-20s] %d%% " % ('-'*int(20*j), 100*j))
        sys.stdout.flush()
        
        # Randomise the initial parameter starting point for those we are fitting (first iteration will just use our guesses)
        for p in range(len(fitwhichparams)):
            if fitwhichparams[p]==1:
                initparams[p] = random.uniform(lowerbound[p], upperbound[p])
        
    end = time.time()
    print("\nOptimisation complete ({:} iterations, {:.0f} s)".format(numiter,end-start))
    return fitted_params, fitted_SSE

#------------------------------------

def fitNFoldModels(maxiter, folds, z, otherparams, plotFLAG=True):
    # This function will fit n-fold symmetry lattice models to the data in z, given all the other parameters (otherparams) except angle and phase.
    # Choose which n-fold models to test by specifying a list 'folds' e.g. [4,6,7,8]
    # Fix the other parameters listed in otherparams (and provide a starting guess for the angle and phase) e.g. [3, 3, 90, 0.4, 0, 0]
    # Lattice parameters: gridAspacing, gridBspacing, alpha, noise, angle, phase
    # Note: the third parameter input to otherparams is meaningless because we fix this by the n-fold.
    
    nfoldSSE = nanArray((len(folds),1))
    nfoldParams = nanArray((len(folds),len(otherparams)))
    
    for i in range(len(folds)):
        testfold = folds[i]
        
        # fit to the angle for this fold of symmetry
        alpha = 360/testfold
        guess_params = [otherparams[i] for i in range(len(otherparams))]
        guess_params[2] = alpha
        fitwhichparams=[0,0,0,0,1,1]
        fitted_params, fitted_SSE = fitLattice(maxiter, z, fitwhichparams, guess_params)

        # choose the best fit for this fold
        opt_iter = np.nanargmin(fitted_SSE)
        opt_params = fitted_params[opt_iter][:]
        nfoldSSE[i] = fitted_SSE[opt_iter]
        nfoldParams[i] = opt_params
        str_params = ''.join("{:.2f}  ".format(e) for e in opt_params)
        
        print("Fitted params for {:}-fold symmetry: ".format(testfold)+str_params)
        print("SSE: {:}".format(nfoldSSE[i]))
        print("=======================")
    
    bestFold = np.argmin(nfoldSSE)
    rankedFolds = ((np.argsort(nfoldSSE, axis=0)).flatten()).tolist()
    print("Best n-fold models:")
    for i in range(len(rankedFolds)):
        print("{:}.  {:}-fold symmetry (SSE={:})".format(i+1, folds[rankedFolds[i]], nfoldSSE[rankedFolds[i]]))
    print("=======================")
    
    if plotFLAG:
        plt.figure(figsize=(3,3))
        plt.plot(folds, nfoldSSE, 'x', color='blue', markeredgewidth=2, ms=10)
        plt.plot(folds[bestFold], nfoldSSE[bestFold], 'x', color='deeppink', markeredgewidth=3, ms=12)
        plt.ylabel('SSE of n-fold model fit')
        plt.xlabel('n-fold symmetry model')
        ax = plt.gca()
        ax.set_xlim([1,10])
        ax.set_ylim([np.min(nfoldSSE)*0.95,np.max(nfoldSSE)*1.05])
    
    return nfoldParams, nfoldSSE
#------------------------------------

def visualizeParameterFit(whichParam, fitted_params, fitted_SSE):
    # Evaluate the fitting surface for a given parameter
    plt.figure(figsize=(2.5,3))
    p = [fitted_params[i][whichParam] for i in range(len(fitted_params))]
    SSE = [fitted_SSE[i] for i in range(len(fitted_SSE))]
    plt.plot(p, SSE, 'x')
    plt.title("Fitting surface (each 'x' is one optimisation)")
    plt.ylabel("SSE")
    plt.xlabel("Parameter value")
    ax = plt.gca()
    return ax

#------------------------------------

def visualCompareGridModels(x, y, z, popt, gridrange, plotrange):
    # Plot the actual data, and next to it data generated from the fitted model
    # 'popt' is a list of parameters for each model ie [nmodels x nparams]
    nmodels = len(popt)
    f, ax = plt.subplots(1, nmodels+1)

    # Actual data
    plotGrid(ax[0], x, y, z, plotrange, "Actual data", False) 
    ax[0].set_aspect('equal', adjustable='box')
    ax[0].set_title("Actual data")
    resolution = z.shape[0]

    # Fitted model
    for i in range(nmodels):
        xfit,yfit,zfit = createGrid(*popt[i], resolution, gridrange, plotrange)
        plotGrid(ax[i+1], xfit, yfit, zfit, plotrange, "Fitted", False) 
        ax[i+1].set_aspect('equal', adjustable='box')
        ax[i+1].set_title("Fitted model {:}".format(i+1))
        ax[i+1].axis('off')
    return ax
#------------------------------------

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

#------------------------------------

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
            >>> angle_between((1, 0, 0), (0, 1, 0))
        Source: https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python/13849249#13849249
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

#------------------------------------

def aggregateBresenhamLine(z, start, end):
    """ Use the Bresenham Line Drawing Algorithm to determine which cells on the path from x1,y1 to x2,y2
    should be considered 'traversed'. The activity in matrix z of the chosen cell indices is integrated
    to form an aggregate across the 'path'.
    
    Edited from source: http://itsaboutcs.blogspot.com/2015/04/bresenhams-line-drawing-algorithm.html """
    
    aggregateActivity = 0
    nCellsTraversed = 0
    drawing = np.copy(z)
    drawing.fill(0)
    
    x1, y1 = start
    x2, y2 = end
    
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    slope = dy/float(dx)
    
    x, y = x1, y1   
 
    # checking the slope if slope > 1 
    # then interchange the role of x and y
    if slope > 1:
        dx, dy = dy, dx
        x, y = y, x
        x1, y1 = y1, x1
        x2, y2 = y2, x2
 
    # initialization of the inital disision parameter
    p = 2 * dy - dx
    
    # Record the activity at this point and increment cell counter
    aggregateActivity += z[x,y]
    nCellsTraversed += 1
    drawing[x,y] = 1

    for k in range(2, dx):
        if p > 0:
            y = y + 1 if y < y2 else y - 1
            p = p + 2*(dy - dx)
        else:
            p = p + 2*dy
 
        x = x + 1 if x < x2 else x - 1
        
        # Record the activity at this point and increment cell counter
        aggregateActivity += z[x,y]
        nCellsTraversed += 1
        drawing[x,y] = 1
    
    return aggregateActivity, nCellsTraversed, drawing

#------------------------------------