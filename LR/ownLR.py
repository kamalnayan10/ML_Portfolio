"""
Creating a simple Linear regression algorithm from scratch
"""

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use("fivethirtyeight")

def create_dataset(num_points , variance , step = 2, correlation = False):

    val = 1
    y = []
    for i in range(num_points):
        gen_val = val+random.randrange(-variance , variance)
        y.append(gen_val)
        if correlation and correlation == "pos":
            val += step
        elif correlation and correlation == "neg":
            val -= step

    x = [i for i in range(len(y))]

    return np.array(x , dtype = np.float64) , np.array(y , dtype = np.float64)

# REGRESSION LINE FUNCTION
def best_fit_slope_and_intercept(x , y):
    m = ( ( (mean(x)*mean(y)) - mean(x*y) ) /
         ( (mean(x)**2) - mean(x**2) ) 
         )  
    
    b = mean(y) - m*mean(x)
    
    return m , b


def squared_error(y_orig , y_line):
    return sum((y_line - y_orig)**2)

def coeff_of_determination(y_orig , y_line):
    y_mean_line = [mean(y_orig) for y in y_orig]
    squared_error_regr = squared_error(y_orig , y_line)

    squared_error_y_mean = squared_error(y_orig , y_mean_line)

    return(1 - (squared_error_regr/ squared_error_y_mean))



# SAMPLE DATA
# xs = np.array([1,2,3,4,5,6],dtype = np.float64)
# ys = np.array([5,4,6,5,6,7],dtype = np.float64)

xs , ys = create_dataset(40 , 10 , 2 , correlation="pos")
print(ys)

m,b= best_fit_slope_and_intercept(xs , ys)

regression_line = [(m*x)+b for x in xs]

# PREDICTION
predict_x = 8
predict_y = (m*predict_x)+b

# R_SQUARED ERROR
r_squared = coeff_of_determination(ys , regression_line)
print(r_squared)

plt.scatter(xs , ys)
# plt.scatter(predict_x , predict_y, color="g")
plt.plot(xs , regression_line)
plt.show()