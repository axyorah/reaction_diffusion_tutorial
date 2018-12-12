import numpy as np
from tqdm import tqdm_notebook as tqdm

def euler_forward(dxdt, x, t_span, p):
    """returns the values of x at time instances given in t_span 
    calculated using function dxdt with parameters p
    INPUTS:
        dxdt: function object: function that returns dxdt as flat 1D ndarray
        x: 1D array:           initial condition
        t_span: 1D array:      time instances
        p: dict:               parameters used by dxdt
    OUTPUT:
        x_span: (len(x), len(t_span)) array"""
    
    x_span = np.zeros((len(x), len(t_span)))
    x_span[:,0] = x
    
    for it,t in enumerate(tqdm(t_span[1:])): # we already know x at t=0
        
        dt = t_span[it+1] - t_span[it]
        x += dxdt(t, x, p)*dt
        x_span[:,it+1] = x
                
    return x_span 

def runge_kutta2(dxdt, x, t_span, p):
    """returns the values of x at time instances given in t_span 
    calculated using function dxdt with parameters p
    INPUTS:
        dxdt: function object: function that returns dxdt as flat 1D ndarray
        x: 1D array:           initial condition
        t_span: 1D array:      time instances
        p: dict:               parameters used by dxdt
    OUTPUT:
        x_span: (len(x), len(t_span)) array"""
    x_span = np.zeros((len(x), len(t_span)))
    x_span[:,0] = x
        
    for it,t in enumerate(tqdm(t_span[1:])):
        dt = t_span[it+1] - t_span[it]
        k1 = dxdt(t,        x,        p)*dt
        k2 = dxdt(t+0.5*dt, x+0.5*k1, p)*dt
        x += k2
        x_span[:,it+1] = x
        
    return x_span

def convert2img(x, size, steepness=5.5, midpoint=0.5):
    """transforms 1D array x into an image of given size
    INPUTS:
        x: 1D array [float]: flat array containing color info for each pixel
        size: (height, width, #channels) [int]: size of the image
        steepness: float: steepness of logistic function at midpoint (used for transforming x)
        midpoint: float: midpoint of logistic function (used for transforming x)
    OUTPUTS:
        (height, width, 4) array of ints ranging from 0 to 255 
        (first 3 depth layers are the RGB channels, and the last layer is the transparency layer)
        """        
    # reshape
    x = x.reshape(size)
    
    # rescale into (0,1) floats
    #x = np.clip(x, 0, 1)
    x = 1/(1 + np.exp(-steepness*(x-midpoint)))        
    
    # rescale into (0,255) ints
    x *= 255
    x = x.astype(np.uint8)
    
    # add/remove color channels
    # if x has less than 3 depth layers:
    if size[2] < 3:
        # add dummy channels filled with 255//2
        x = np.concatenate((x, 255//2 * np.ones((size[0], size[1], 3-size[2]), dtype=np.uint8)), axis=2)
            
    # if x has more than 4 depth layers - trim all layers after 4th
    # (first 3 layers would be interpreted as RGB channels and 4th layer - as transparency channel)
    if size[2] > 4:
        x = x[:,:,:3]    
    
    # if x has exactly 3 depth layers:
    if x.shape[2] == 3:
        # add transparency channel and set each val to 255 (no transparency)
        x = np.concatenate((x, 255*np.ones((size[0], size[1], 1), dtype=np.uint8)), axis=2)
        
    return  x