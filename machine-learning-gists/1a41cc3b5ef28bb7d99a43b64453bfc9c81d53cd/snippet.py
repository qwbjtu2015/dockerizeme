'''
This script is based on the original work of Randal S. Olson (randalolson.com) for the Traveling Salesman Portrait project.
http://www.randalolson.com/2018/04/11/traveling-salesman-portrait-in-python/

Please check out the original project repository for information:

https://github.com/rhiever/Data-Analysis-and-Machine-Learning-Projects

The script was updated by Joshua L. Adelman, adapting the work of Antonio S. Chinchón described in the following blog post:
https://fronkonstin.com/2018/04/17/pencil-scribbles/

This code is released under an MIT License (https://opensource.org/licenses/MIT):

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files 
(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, 
publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, 
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR 
ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH 
THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

import os
import math
import matplotlib.pyplot as plt
import numpy as np
import urllib.request
from PIL import Image
from tsp_solver.greedy_numpy import solve_tsp
from scipy.spatial.distance import pdist, squareform

import time

image_url = 'http://ereaderbackgrounds.com/movies/bw/Frankenstein.jpg'
image_path = 'Frankenstein.jpg'

n_samples = 400
n_iter = 250
use_grayscale = True  # Use grayscale

out_path = 'traveling-salesman-portrait-pencil-{}.png'.format('gs' if use_grayscale else 'bw')

if not os.path.exists(image_path):
    urllib.request.urlretrieve(image_url, image_path)

original_image = Image.open(image_path)

if use_grayscale:
    bw_image = original_image.convert('L', dither=Image.NONE)
    bw_image_array = np.array(bw_image, dtype=np.float64)
    black_indices = np.argwhere(bw_image_array >= 0)
    P = (1.0 - bw_image_array.flatten() / 255.0)
else:
    bw_image = original_image.convert('1', dither=Image.NONE)
    bw_image_array = np.array(bw_image, dtype=np.int)
    black_indices = np.argwhere(bw_image_array == 0)
    P = np.ones(black_indices.shape[0], dtype=np.float64)

P = P / P.sum()

plt.figure(figsize=(8, 10), dpi=600)

t1 = time.time()
t2 = time.time()
for i in range(n_iter):
    print(i, t2-t1)
    t1 = time.time()
    chosen_black_indices = black_indices[np.random.choice(black_indices.shape[0], replace=False, size=n_samples, p=P)]

    distances = pdist(chosen_black_indices)
    distance_matrix = squareform(distances)

    optimized_path = solve_tsp(distance_matrix)

    optimized_path_points = [chosen_black_indices[x] for x in optimized_path]

    alpha = np.random.uniform(0.0, 0.1)
    plt.plot([x[1] for x in optimized_path_points], [x[0] for x in optimized_path_points], color='black', lw=1, alpha=alpha)
    t2 = time.time()

plt.xlim(0, 600)
plt.ylim(0, 800)
plt.gca().invert_yaxis()
plt.xticks([])
plt.yticks([])
plt.savefig(out_path, bbox_inches='tight')