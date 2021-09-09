from neorl.benchmarks.tsp import TSP
from neorl.benchmarks.kp import KP

def bench_2dplot(function, domain=(-100,100), points=30, savepng=None):
    """
    Creates a 2D surface plot of a function.

    Args:
        function (function): The objective function to be called at each point.
        domain (num, num): The inclusive (min, max) domain for each dimension.
        points (int): The number of points to discretize in x and y dimensions.
        savepng (str): save png file for the plot
    """
    from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 
    import matplotlib.pyplot as plt
    import numpy as np
    try:
        
        plt.figure()
        dimension=2
        
        # create points^2 tuples of (x,y) and populate z
        xys = np.linspace(domain[0], domain[1], points)
        xys = np.transpose([np.tile(xys, len(xys)), np.repeat(xys, len(xys))])
        zs = np.zeros(points*points)
    
        if dimension > 2:
            # concatenate remaining zeros
            tail = np.zeros(dimension - 2)
            for i in range(0, xys.shape[0]):
                zs[i] = function(np.concatenate([xys[i], tail]))
        else:
            for i in range(0, xys.shape[0]):
                zs[i] = function(xys[i])
    
        # create the plot
        ax = plt.axes(projection='3d')
    
        X = xys[:,0].reshape((points, points))
        Y = xys[:,1].reshape((points, points))
        Z = zs.reshape((points, points))
        ax.plot_surface(X, Y, Z, cmap='ocean', edgecolor='none')
        ax.set_title(function.__name__)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        
        if savepng:
            plt.savefig(savepng, dpi=200, format='png')
            
        plt.show()
        
    except:
        raise Exception ('--error: Plotting fails, if you use CEC17, then f11-f20, f29, f30 are not defined for d=2 dimensions')