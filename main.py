import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Runge-Kutta 4th Order with Adaptive Step Size
def runge_kutta_4_adaptive(f, x0, y0, h, n, tol=1e-6):
    """
    Solves the ODE using the 4th-order Runge-Kutta method with adaptive step size.

    Parameters:
    f: function representing the derivative dy/dx = f(x, y)
    x0: initial x value
    y0: initial y value
    h: initial step size
    n: maximum number of steps
    tol: tolerance for adaptive step size

    Returns:
    Tuple of lists (x, y) containing x and y values.
    """
    
    def step(xn, yn, h, remaining_steps):
        """Recursive helper function for adaptive step size Runge-Kutta."""
        if remaining_steps == 0:
            return [xn], [yn]
        
        # Runge-Kutta 4th order increments
        k1 = h * f(xn, yn)
        k2 = h * f(xn + h/2, yn + k1/2)
        k3 = h * f(xn + h/2, yn + k2/2)
        k4 = h * f(xn + h, yn + k3)
        y_next = yn + (k1 + 2*k2 + 2*k3 + k4) / 6
        
        # Estimate error using a half-step
        h_half = h / 2
        k1_half = h_half * f(xn, yn)
        k2_half = h_half * f(xn + h_half/2, yn + k1_half/2)
        k3_half = h_half * f(xn + h_half/2, yn + k2_half/2)
        k4_half = h_half * f(xn + h_half, yn + k3_half)
        y_half = yn + (k1_half + 2*k2_half + 2*k3_half + k4_half) / 6

        # Calculate error and adjust step size
        error_estimate = np.abs(y_half - y_next)
        if error_estimate > tol:
            return step(xn, yn, h * 0.5, remaining_steps)
        elif error_estimate < tol / 10:
            h *= 2.0
        
        # Proceed to the next step
        x_next = xn + h
        xs, ys = step(x_next, y_next, h, remaining_steps - 1)
        return [xn] + xs, [yn] + ys

    return step(x0, y0, h, n)


# Backward Euler Method for Stiff ODEs
def backward_euler(f, x0, y0, h, n):
    """
    Solves the ODE using the implicit backward Euler method.

    Parameters:
    f: function representing the derivative dy/dx = f(x, y)
    x0: initial x value
    y0: initial y value
    h: step size
    n: number of steps

    Returns:
    Tuple of lists (x, y) containing x and y values.
    """

    def step(xn, yn, remaining_steps):
        """Recursive helper function for the backward Euler method."""
        if remaining_steps == 0:
            return [xn], [yn]

        # Implicit equation to solve for y_next
        func = lambda yn_next: yn_next - yn - h * f(xn + h, yn_next)
        y_next = fsolve(func, yn)[0]
        
        # Proceed to the next step
        x_next = xn + h
        xs, ys = step(x_next, y_next, remaining_steps - 1)
        return [xn] + xs, [yn] + ys

    return step(x0, y0, n)


# Solving Higher-Dimensional Systems of ODEs
def system_of_odes_high_dim(F, x0, y0, h, n):
    """
    Solves a higher-dimensional system of first-order ODEs using the 4th-order Runge-Kutta method.

    Parameters:
    F: list of functions representing the system dy/dx = F(x, Y)
    x0: initial x value
    y0: list of initial y values
    h: step size
    n: number of steps

    Returns:
    Tuple of lists (x, Y) where x is the list of x values, and Y is a list of lists for y values of each function.
    """

    def step(xn, Yn, remaining_steps):
        """Recursive helper function for solving the system of ODEs."""
        if remaining_steps == 0:
            return [xn], [Yn]

        # Runge-Kutta 4th order for each equation in the system
        def rk4_step(f, Yn):
            k1 = h * f(xn, Yn)
            k2 = h * f(xn + h/2, Yn + k1/2)
            k3 = h * f(xn + h/2, Yn + k2/2)
            k4 = h * f(xn + h, Yn + k3)
            return Yn + (k1 + 2*k2 + 2*k3 + k4) / 6
        
        # Calculate the next Y values for each function in F
        Y_next = np.array([rk4_step(f, Yn) for f in F])
        
        # Proceed to the next step
        x_next = xn + h
        xs, Ys = step(x_next, Y_next, remaining_steps - 1)
        return [xn] + xs, [Yn] + Ys

    return step(x0, np.array(y0), n)


# Example ODE functions
def example_function(x, y):
    return x - y

def stiff_function(x, y):
    return -1000 * y + 3000 - 2000 * np.exp(-x)

# Example functions for a higher-dimensional system
def f1(x, Y):
    return Y[1] - Y[0] * np.cos(x)

def f2(x, Y):
    return -Y[0] * np.sin(x) + Y[1] * np.cos(x)


# Example usage: Adaptive Runge-Kutta Method
x0, y0, h, n = 0, 1, 0.1, 100
x_adaptive, y_adaptive = runge_kutta_4_adaptive(example_function, x0, y0, h, n)
plt.plot(x_adaptive, y_adaptive, label="Runge-Kutta 4th Order with Adaptive Step Size")
plt.xlabel('x')
plt.ylabel('y')
plt.title("ODE Solution using Adaptive Runge-Kutta Method")
plt.legend()
plt.show()

# Example usage: Backward Euler Method for Stiff ODE
x0, y0, h, n = 0, 0, 0.001, 1000
x_stiff, y_stiff = backward_euler(stiff_function, x0, y0, h, n)
plt.plot(x_stiff, y_stiff, label="Backward Euler Method (Stiff ODE)")
plt.xlabel('x')
plt.ylabel('y')
plt.title("Stiff ODE Solution using Backward Euler Method")
plt.legend()
plt.show()

# Example usage: Higher-Dimensional System of ODEs
x0, y0, h, n = 0, [1, 0], 0.1, 100
x_high_dim, Y_high_dim = system_of_odes_high_dim([f1, f2], x0, y0, h, n)
plt.plot(x_high_dim, [Y[0] for Y in Y_high_dim], label="y1")
plt.plot(x_high_dim, [Y[1] for Y in Y_high_dim], label="y2")
plt.xlabel('x')
plt.ylabel('y1, y2')
plt.title("Higher-Dimensional System of ODEs Solution using Runge-Kutta Method")
plt.legend()
plt.show()
