# Copyright 2018 University of Basel, Center for medical Image Analysis and Navigation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch as th

import airlab.utils.image as image

def create_C_2_O_test_images(image_size, dtype=th.float32, device='cpu'):
    x = np.linspace(-1, 1, image_size)

    xv, yv = np.meshgrid(x, x)

    value = xv**2 + yv**2

    index = (value > 0.33**2) & (value < 0.64**2)

    fixed_image = np.ones((image_size, image_size))
    fixed_image[index] = 0

    index = (xv > 0) & (np.abs(yv) < 0.16)
    fixed_image[index] = 1

    moving_image = 0.5**2 - xv**2 - yv**2

    shaded_image = moving_image.copy()*10
    index = moving_image < 0
    shaded_image[index] = 1

    shaded_image[np.logical_not(index)] = shaded_image[np.logical_not(index)]/np.max(shaded_image[np.logical_not(index)])



    index_2 = moving_image > 0
    moving_image[index_2] = 0

    moving_image[index] = 1


    return [image.image_from_numpy(fixed_image, [1, 1], dtype=dtype, device=device),
            image.image_from_numpy(moving_image, [1, 1], dtype=dtype, device=device),
            image.image_from_numpy(shaded_image, [1, 1], dtype=dtype, device=device)]


