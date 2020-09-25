import os
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def load_points_from_file(filename):
    """Loads 2d points from a csv called filename
    Args:
        filename : Path to .csv file
    Returns:
        (xs, ys) where xs and ys are a numpy array of the co-ordinates.
    """
    points = pd.read_csv(filename, header=None)
    return points[0].values, points[1].values


def view_data_segments(xs, ys):
    """Visualises the input file with each segment plotted in a different colour.
    Args:
        xs : List/array-like of x co-ordinates.
        ys : List/array-like of y co-ordinates.
    Returns:
        None
    """
    assert len(xs) == len(ys)
    assert len(xs) % 20 == 0
    len_data = len(xs)
    num_segments = len_data // 20
    colour = np.concatenate([[i] * 20 for i in range(num_segments)])
    plt.set_cmap('Dark2')
    plt.scatter(xs, ys, c=colour)
    plt.show()


# separate the data into respective line segments of 20 points
def line_segments(xs, ys):
    num_segments = len(xs) // 20
    xss = np.array(np.split(xs, num_segments))
    yss = np.array(np.split(ys, num_segments))
    return xss, yss


# function to calculate least square regression of linear fnx
def least_squares_linear(xs, ys):
    ones = np.ones(xs.shape)
    x = np.column_stack((ones, xs))
    a_l_s = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(ys)
    return a_l_s


# function to calculate least square regression of polynomial fnx of degree k
def least_squares_polynomial(xs, ys, k):
    ones = np.ones(xs.shape)
    dummy = xs.copy()
    for i in range(2, k + 1):
        xs_k = dummy ** i
        xs = np.column_stack((xs, xs_k))
    x = np.column_stack((ones, xs))
    a_l_s = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(ys)
    return a_l_s


# function to calculate least square regression of sine function
def least_square_sine(xs, ys):
    ones = np.ones(xs.shape)
    x = np.column_stack((ones, np.sin(xs)))
    a_l_s = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(ys)
    return a_l_s


# function that computes SSE of linear function
def square_error_linear(xs, ys, a, b):
    ys_hat = a + b * xs
    return np.sum((ys - ys_hat) ** 2)


# function that computes SSE of polynomial functions with degree 3
def square_error_poly3(xs, ys, a, b, c, d):
    ys_hat = a + (b * xs) + (c * (xs ** 2)) + (d * (xs ** 3))
    return np.sum((ys - ys_hat) ** 2)


# function that computes SSE of sine functions
def square_error_sine(xs, ys, a, b):
    ys_hat = a + b * (np.sin(xs))
    return np.sum((ys - ys_hat) ** 2)


# returns all the Least square regression in a tuple
def calc_least_square(xs, ys):
    a = least_squares_linear(xs, ys)
    b = least_squares_polynomial(xs, ys, 3)
    c = least_square_sine(xs, ys)
    return a, b, c


# Determine index  of value with minimal residual error  hence best line of fit
def find_line_min_error(xs, ys):
    xss, yss = line_segments(xs, ys)
    num_segments = len(xs) // 20
    min_index = np.zeros((num_segments, 1))
    sq_error = np.zeros((3, 1))
    for i in range(num_segments):
        a, b, c = calc_least_square(xss[i], yss[i])
        sq_error[0] = square_error_linear(xss[i], yss[i], a[0], a[1])
        sq_error[1] = square_error_poly3(xss[i], yss[i], b[0], b[1], b[2], b[3])
        sq_error[2] = square_error_sine(xss[i], yss[i], c[0], c[1])
        min_index[i] = np.argmin(sq_error)
    return min_index


# Sum of all the Square errors of each line segments
def total_reconstruction_error(xs, ys):
    xss, yss = line_segments(xs, ys)
    num_segments = len(xs) // 20
    sq_error = np.zeros((3, 1))
    list_error = np.zeros((num_segments, 1))
    for i in range(num_segments):
        a, b, c = calc_least_square(xss[i], yss[i])
        sq_error[0] = square_error_linear(xss[i], yss[i], a[0], a[1])
        sq_error[1] = square_error_poly3(xss[i], yss[i], b[0], b[1], b[2], b[3])
        sq_error[2] = square_error_sine(xss[i], yss[i], c[0], c[1])
        list_error[i] = sq_error[np.argmin(sq_error)]
    total_sum = np.sum(list_error)
    print(total_sum, '\n')


# visualize the line segments plotted with their regression variables
def visualize_points(xs, ys):
    indexes = find_line_min_error(xs, ys)
    num_segments = len(xs) // 20
    xss, yss = line_segments(xs, ys)
    fig, ax = plt.subplots()
    colour = np.concatenate([[i] * 20 for i in range(num_segments)])
    ax.scatter(xss, yss, s=100, c=colour)
    for i in range(num_segments):
        x_1_1r = xss[i].min()
        x_1_2r = xss[i].max()
        if indexes[i] == 0:
            a = least_squares_linear(xss[i], yss[i])
            y_1_1r = a[0] + a[1] * x_1_1r
            y_1_2r = a[0] + a[1] * x_1_2r
            ax.plot([x_1_1r, x_1_2r], [y_1_1r, y_1_2r], 'r-', lw=4)
        elif indexes[i] == 1:
            new_x = np.linspace(x_1_1r, x_1_2r, len(xs))
            a = least_squares_polynomial(xss[i], yss[i], 3)
            new_y = a[0] + a[1] * new_x + a[2] * new_x ** 2 + a[3] * new_x ** 3
            ax.plot(new_x, new_y, 'r-', lw=4)
        else:
            new_x = np.linspace(x_1_1r, x_1_2r, len(xs))
            a = least_square_sine(xss[i], yss[i])
            new_y = a[0] + a[1] * np.sin(new_x)
            ax.plot(new_x, new_y, c='r')

    plt.show()


# Output the Total reconstruction error or visualize the line segments
def main():
    filename = sys.argv[1]
    xs, ys = load_points_from_file(filename)
    if len(sys.argv) == 3 and sys.argv[2] == "--plot":
        total_reconstruction_error(xs, ys)
        visualize_points(xs, ys)

    else:
        total_reconstruction_error(xs, ys)


main()
