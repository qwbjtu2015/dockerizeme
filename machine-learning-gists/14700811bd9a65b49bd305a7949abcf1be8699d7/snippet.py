"""
Demo code for showing how a Self-Organized Map (Kohonen Map) learns handwritten digit data in an
unsupervised manner.

Data source: https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/

This code uses 'optdigits.tra'
Each line is a handwritten digit.
The first 64 numbers are gray levels for an 8 x 8 image.
The last number is the digit class.


The SOM is represented as a N x M x D array of floats. With N x M being the sheet of SOM elements and
D being the dimensionality of the input and thus the size of the weight vector for each element.
"""
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

import numpy as np


def load_data(fname='optdigits.tra'):
  raw_data = np.loadtxt(fname, dtype=int, delimiter=',')
  image_data = (raw_data[:, :64] - 8) / 8.0
  class_data = raw_data[:, 64]
  print(Counter(class_data))
  return image_data, class_data


def update_som(som, x, sigma, eta):
  """Given an input pattern, update the SOM weights

  :param som:  N x M x D array of floats. With N x M being the sheet of SOM elements
               and D being the dimensionality of the input and thus the size of the
               weight vector for each element.
  :param x:    D element vector representing on handwritten image
  :param sigma: size of neighborhood
  :param eta:
  :return:
  """
  wd = som - x
  iw, jw = np.unravel_index(np.argmin((wd ** 2).sum(axis=2)), som.shape[:2])

  for i in range(som.shape[0]):
    for j in range(som.shape[1]):
      N = np.exp(-((i - iw) ** 2 + (j - jw) ** 2) / (2 * sigma ** 2))
      som[i, j, :] -= eta * N * wd[i, j, :]


def visualize_som_as_classes(som, image_data, class_data):
  """Given a SOM, and a list of inputs, count how many times a sample from a class wins at a certain SOM
  element and plot those class values with a font size proportional to the winning counts.

  :param som:
  :param image_data:
  :param class_data:
  :return:
  """
  counts = np.zeros(som.shape[:2] + (10,), dtype=int)
  for id, ic in zip(image_data, class_data):
    iw, jw = np.unravel_index(np.argmin(((som - id) ** 2).sum(axis=2)), som.shape[:2])
    counts[jw, iw, ic] += 1

  counts = counts.astype(float) / counts.sum()
  for i in range(som.shape[0]):
    for j in range(som.shape[1]):
      for c in range(10):
        plt.text(j, som.shape[0] - i - 1, c, fontdict={'size': 400 * counts[j, i, c]})

  plt.setp(plt.gca(), xlim=(-1, som.shape[1]), ylim=(-1, som.shape[0]), xticks=[], yticks=[])


def visualize_som_as_templates(som):
  """Given a SOM plot the templates represented by each SOM element.

  :param som:
  :return:
  """
  plt.subplots_adjust(wspace=0, hspace=0)
  for i in range(som.shape[0]):
    for j in range(som.shape[1]):
      plt.subplot(som.shape[0], som.shape[1], j + i * som.shape[1] + 1)
      plt.imshow(som[i, j, :].reshape(8, 8), cmap=plt.cm.gray_r, interpolation='none')
      plt.setp(plt.gca(), xticks=[], yticks=[])


def animate_som_learning(sigma=4.0, sigma_k=0.9, eta=0.001, eta_k=0.8, iters=2000, view_type='class'):
  """Initialize a SOM and run it, plotting it's state every few iterations

  :param sigma:
  :param eta:
  :param iters:
  :param view_type: 'class' or 'template'
  :return:
  """
  FFMpegWriter = manimation.writers['ffmpeg']
  metadata = dict(title='Movie Test', artist='Matplotlib',
                  comment='Movie support!')
  writer = FFMpegWriter(fps=15, metadata=metadata)

  image_data, class_data = load_data(fname='optdigits.tra')
  np.random.seed(17)
  som = np.random.randn(10, 10, 64)
  fig = plt.figure(figsize=(5, 5))

  with writer.saving(fig, "som_digits_{}.mp4".format(view_type), 150):
    last_classes = []
    for i in range(iters):
      if i % 10 == 0:
        if view_type == 'class':
          visualize_som_as_classes(som, image_data, class_data)
        else:
          visualize_som_as_templates(som)
        plt.suptitle('sigma = {:0.2f}, eta = {:0.2f}\nlast data = {}'.format(sigma, eta, last_classes), fontsize=9)
        writer.grab_frame()
        plt.cla()
        last_classes = []

      update_som(som, image_data[i, :], sigma, eta)
      sigma *= sigma_k
      eta *= eta_k
      last_classes.append(class_data[i])
      print(i, sigma, eta)


if __name__ == '__main__':
  iters = 2000
  sigma = 2.0
  sigma_k = 1.0
  eta = 0.1
  eta_k = 0.999
  if len(sys.argv) < 2:
    print('python {} "class"/"template"'.format(sys.argv[0]))
    exit(0)
  animate_som_learning(
    sigma=sigma, sigma_k=sigma_k,
    eta=eta, eta_k=eta_k,
    iters=iters,
    view_type=sys.argv[1]
  )