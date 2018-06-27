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

import torch as th

import matplotlib.pyplot as plt
import time

from airlab.registration import registration as al
import airlab.utils.image as imu
import airlab.transformation.pairwiseTransformation as pt
import airlab.transformation.utils as tu
import airlab.loss.pairwiseImageLoss as pil
import airlab.regulariser.demonsRegulariser as dreg

import create_test_image_data as test_data

def main():
    start = time.time()

    # set the used data type
    dtype = th.float32
    # set the device for the computaion to CPU
    device = th.device("cpu")

    # In order to use a GPU uncomment the following line. The number is the device index of the used GPU
    # Here, the GPU with the index 0 is used.
    device = th.device("cuda:0")

    fixed_image, moving_image, shaded_image = test_data.create_C_2_O_test_images(256,
                                                                                 dtype=dtype, device=device)

    # create pairwise registration object
    registration = al.DemonsRegistraion(dtype=dtype, device=device)

    # choose the affine transformation model
    transformation = pt.NonParametricTransformation(moving_image.size, dtype=dtype, device=device)

    registration.set_transformation(transformation)

    # choose the Mean Squared Error as image loss
    image_loss = pil.MSELoss(fixed_image, moving_image)

    registration.set_image_loss([image_loss])

    # choose a regulariser for the demons
    regulariser = dreg.GaussianRegulariser(moving_image.spacing, sigma=[2, 2], dtype=dtype, device=device)

    registration.set_regulariser([regulariser])

    # choose the Adam optimizer to minimize the objective
    optimizer = th.optim.Adam(transformation.parameters(), lr=0.07)

    registration.set_optimizer(optimizer)
    registration.set_number_of_iterations(1000)

    # start the registration
    registration.start()

    # warp the moving image with the final transformation result
    displacement = transformation.get_displacement()


    #use the shaded version of the fixed image for visualization
    # itkImg = sitk.ReadImage("./data/circle_moving_image_shaded_2d.png", sitk.sitkFloat32)
    # itkImg = sitk.RescaleIntensity(itkImg, 0, 1)
    # moving_image_shaded = imu.create_tensor_image_from_itk_image(itkImg, dtype=dtype, device=device)
    warped_image = tu.warp_image(shaded_image, displacement)

    end = time.time()

    displacement = imu.create_displacement_image_from_image(displacement, moving_image)

    print("=================================================================")

    print("Registration done in: ", end - start)

    # plot the results
    plt.subplot(221)
    plt.imshow(fixed_image.numpy(), cmap='gray')
    plt.title('Fixed Image')

    plt.subplot(222)
    plt.imshow(moving_image.numpy(), cmap='gray')
    plt.title('Moving Image')

    plt.subplot(223)
    plt.imshow(warped_image.numpy(), cmap='gray')
    plt.title('Warped Moving Image')

    plt.subplot(224)
    plt.imshow(displacement.magnitude().numpy(), cmap='jet')
    plt.title('Magnitude Displacemente')

    plt.show()

    # write result images
    # sitk.WriteImage(warped_image.itk(), '/tmp/demons_warped_image.vtk')
    # sitk.WriteImage(moving_image.itk(), '/tmp/demons_moving_image.vtk')
    # sitk.WriteImage(fixed_image.itk(), '/tmp/demons_fixed_image.vtk')
    # sitk.WriteImage(shaded_image.itk(), '/tmp/demons_shaded_image.vtk')
    # sitk.WriteImage(displacement.itk(), '/tmp/demons_displacement_image.vtk')


if __name__ == '__main__':
    main()