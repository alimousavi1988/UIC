""" Calculating the UIC(x,y) where x and y each has the format of [n,dim].
n is the number of samples and dim is the dimension of the variable. """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def data():
  """This function provides the data to the UIC algorithm."""

  # put the data generator below. A sample is provided
  ######################################################################

  # shape of variables and number of samples
  size_x = 2
  size_y = 1
  num_samples = 5000

  # Data
  x = np.random.rand(num_samples, size_x)
  y = (np.power((x[:,0] - 0.5),3) + np.power((x[:,1] - 0.5),3)).reshape(num_samples,size_y)

  ######################################################################

  if x.shape != (num_samples, size_x):
    raise ValueError("The Shape of variable X is not correct!") 
  if y.shape != (num_samples, size_y):
    raise ValueError("The Shape of variable Y is not correct!") 
  d = np.concatenate((x, y), axis=1)


  return d, size_x, size_y, num_samples



def UIC(D, size_x, size_y, num_points, grid_factor):
  """the main function for running the UIC algorithm.

  Args:
    D: Dataset
    size_x: diemnsion of x
    size_y: dimension of y
    num_points: number of sample points
    grid_factor: parameter between 0 and 1 (less than 1) denoting the max size of the grid

  Returns:
    final_uic: the UIC value.
  """

  if D.shape != (num_points, size_x + size_y):
    raise ValueError("The Shape of dataset D is not correct!") 

  final_uic = 0.

  for ii in range(2, 1+int((num_points**grid_factor / 2)**(1./size_x))):
    for jj in range(2, 1+int((num_points**grid_factor /
                              ii**size_x)**(1./size_y))):

      x_vec = np.reshape(np.power(ii,
                                  np.linspace(size_x-1, 0, size_x).astype(int)),
                         (size_x, 1))
      y_vec = np.reshape(np.power(jj,
                                  np.linspace(size_y-1, 0, size_y).astype(int)),
                         (size_y, 1))
      xy_vec = np.concatenate((x_vec, y_vec), axis=0).T

      min_vals = np.min(D, axis=0)-0.0001
      max_vals = np.max(D, axis=0)+0.0001

      full_vec = np.concatenate((np.ones((1, size_x)) * ii,
                                 np.ones((1, size_y)) * jj),
                                axis=1)

      unit_vals = np.divide(np.subtract(max_vals, min_vals), full_vec)

      num_units = np.ceil(np.divide(np.subtract(D,
                                                np.tile(min_vals,
                                                        (num_points, 1))),
                                    np.tile(unit_vals, (num_points, 1))))

      num_units = np.multiply(num_units-1, np.tile(xy_vec, (num_points, 1)))

      prob_units = np.concatenate((np.reshape(np.sum(num_units[:, 0:size_x],
                                                     axis=1) + 1,
                                              (num_points, 1)),
                                   np.reshape(np.sum(num_units[:, size_x:],
                                                     axis=1) + 1,
                                              (num_points, 1))), axis=1)

      unique_cells = np.unique(prob_units, return_counts=True, axis=0)
      prob_cell = unique_cells[1] / (1. * num_points)

      unique_row = np.unique(prob_units[:, 0], return_counts=True, axis=0)
      prob_row = unique_row[1] / (1. * num_points)

      unique_col = np.unique(prob_units[:, 1], return_counts=True, axis=0)
      prob_col = unique_col[1] / (1. * num_points)

      h_rowcol = -1. * np.sum(np.multiply(prob_cell, np.log2(prob_cell)))
      h_col = -1. * np.sum(np.multiply(prob_col, np.log2(prob_col)))
      h_row = -1. * np.sum(np.multiply(prob_row, np.log2(prob_row)))

      this_uic = (h_row + h_col - h_rowcol) / (np.log2(np.min((ii**size_x,
                                                               jj**size_y))))

      if this_uic > final_uic:
        final_uic = this_uic

  return final_uic



def main():

  grid_factor = 0.6 # a factor which determinines the max size of the grid.
  D, size_x, size_y, sample_size = data() # getting the data
  best_uic = UIC(D, size_x, size_y, sample_size, grid_factor) # calculating the UIC
  print('UIC value is: {}'.format(best_uic)) # printing the UIC value


if __name__ == '__main__':
  main()

