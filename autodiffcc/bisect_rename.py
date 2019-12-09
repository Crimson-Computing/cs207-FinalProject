from inspect import signature
import matplotlib.pyplot as plt 
import numpy as np
import itertools
  

def _bisect(fun, interval):
    
    
    """Perfroms the bisection method on a function of one or more variables to find a root
    
    INPUTS
    =======
    fun: specified function for example
    interval: list of intervals for the function to look for a root in

    
    RETURNS
    ========
    returns the approximate root 
    
    NOTES
    =====
    This function considers a value to be 0 if round(value, 15) == 0)
    This function does not find all the roots. You can change the intervals to 
    look for different roots. 
    
    EXAMPLES
    =========
    
    >>> def f(x, y):
    >>>    return(2*x*y - 2)
    
    >>> Bisect(f, interval = [[-10, 10], [-4, 10]])
    [0.12999772724065328, 7.692442177460448]
    """
    
    
    print("Please note that this function uses approximations")

    
    # check how many parameters there are in the function
    signa = signature(fun)
    nParam = len(signa.parameters)
    
    # create a point array storing coordinates 
    
    ## example of points 
    #    start-interval finish-interval
    # x       1               2
    # y       3               10
    # z      -2               0
    
    #points = []
    #for i in range(0, nParam): 
       # points.append(interval)
       # print(points)
    points = np.asarray(interval)
    print(points)
        
    #### PLOT if can do 2D graph
    if nParam==1:
        # x axis values 
        a = points[0][0]
        print(a)
        b = points[0][1]
        print(b)
        interval = np.linspace(a, b, num = 1000)
        # corresponding y axis values 
        values = fun(interval)
        # plotting the points  
        plt.plot(interval, values, color='green', linestyle='dashed', linewidth = 0.5, 
                 marker='o', markerfacecolor='blue', markersize=1) 
        # naming the x axis 
        plt.xlabel(' Interval ') 
        # naming the y axis 
        plt.ylabel(' Values ' ) 
        plt.axhline(y=0)
        # giving a title to my graph 
        plt.title('Graphs function in the specified interval')    
        # function to show the plot 
        plt.show() 

    # get the starting points of each variable 
    # get the ending points for each variable
    # get their combinations. Should be 4 different combinations
    
    matrix = np.empty((nParam,2))
    for p in range(0, nParam): 
        matrix[p] = points[p]
    allpoints = list(itertools.product(*matrix))

    # check sign change
    results = []
    for elements in allpoints: 
        results.append(fun(*elements))
        asign = np.sign(results)
        # detect sign change
        signchange = sum(((np.roll(asign, 1) - asign) != 0).astype(int)) > 0
    
    # if signs are not different
    if not signchange:
        raise Exception("No change in sign, please try different intervals")
       
    if signchange: 
        print("Root between in the specified intervals")
        i = 1
        # if signs are different
        while signchange: 
            print("----------------------")
            print("iteration", i)
  
            i = i + 1
            
            # middle points
            c = []
            for k in range(0, nParam):
                c.append((points[k][0]+points[k][1])/2)
        
            middlePointResult = fun(*c)

            # approx to 14ths decimal point
            # if found root: 
            if (round(middlePointResult, 15) == 0):
                print("root found for ", c)
                return(c) # return middle as the approximate root value 
            # if did not find root yet:
            else: 
                j = 0 

                for n in results: 
                    if (n * middlePointResult < 0):
                        print("root between", c, "and", allpoints[j])
                        corner1 = list(allpoints[j])
                        corner2 = c 
                    j=j+1
                print("choosing to look in the area between", corner1, "and", corner2)

                    
                # update points for intervals
                if corner1 > corner2:
                    points = np.c_[corner2,corner1]
                else: 
                    points = np.c_[corner1,corner2]
                    
                points = np.asarray(points)
                
                matrix = np.empty((nParam,2))
                for p in range(0, nParam): 
                    matrix[p] = points[p]
                allpoints = list(itertools.product(*matrix))

                # check sign change
                results = []
                for elements in allpoints: 
                    results.append(fun(*elements))
                    asign = np.sign(results)
                    # detect sign change
                    signchange = sum(((np.roll(asign, 1) - asign) != 0).astype(int)) > 0



def f(x, y):
    return (2 * x * y - 2)

def f(x):
    return (2 * x - 2)

_bisect(f, interval=[[-10, 10]])
