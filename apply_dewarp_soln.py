# This is the script to run for making a warping/astrometric solution to LMIRCam, using data taken
# in Nov and Dev 2016

# parent find_dewarp_soln.py created by E.S., Nov 2016
# child apply_dewarp_soln.py made by E.S., Feb 2017

import numpy as np
from astrom_lmircam_soln import *
from astrom_lmircam_soln import polywarp
from astrom_lmircam_soln import dewarp
from astropy.io import fits
import matplotlib.pyplot as plt


#####################################################################
# SET THE DEWARP COEFFICIENTS

# the below coefficients are relevant for LMIRCam 2048x2048 readouts after modifications in summer 2016
# (in Maire+ 2015 format, here x_0 = y_0 = 0)
'''
Kx = [[ -4.74621436e+00,   9.99369200e-03,  -4.69741638e-06,   4.11937105e-11],
      [  1.01486148e+00,  -2.84066638e-05,   2.10787962e-08,  -3.90558311e-12],
      [ -1.61139243e-05,   2.24876212e-08,  -2.29864156e-11,   6.59792237e-15],
      [  8.88888428e-09,  -1.03720381e-11,   1.05406782e-14,  -3.06854175e-18]]
Ky = [[  9.19683947e+00,   9.84613002e-01,  -1.28813904e-06,   6.26844974e-09],
      [ -7.28218373e-03,  -1.06359740e-05,   2.43203662e-09,  -1.17977589e-12],
      [  9.48872623e-06,   1.03410741e-08,  -2.38036199e-12,   1.17914143e-15],
      [  3.56510910e-10,  -5.62885797e-13,  -5.67614656e-17,  -4.21794191e-20]]

# the below were taken in 2017B (SX ONLY)
Kx = [[ -1.97674665e+01,   2.26890756e-02,  -1.06483884e-05,   1.33174251e-09],
      [  1.04269459e+00,  -2.68457747e-05,   2.08519317e-08,  -4.74786541e-12],
      [ -3.30919802e-05,   9.48855944e-09,  -1.00804780e-11,   3.45894384e-15],
      [  1.00196745e-08,  -2.58289058e-12,   2.58960909e-15,  -8.74827083e-19]]
Ky = [[  1.05428609e+01,   9.91877631e-01,  -1.30947328e-06,   5.98620664e-09],
      [ -2.65330464e-02,  -6.14857421e-06,  -1.56796197e-08,   6.61213303e-12],
      [  1.50777505e-05,  -8.14931285e-09,   2.28968428e-11,  -9.37645995e-15],
      [ -1.54162134e-09,   5.25556977e-12,  -7.46189515e-15,   3.04540450e-18]]

# the below were taken in 2017B (DX ONLY)
Kx = [[ -1.34669677e+01,   2.25398365e-02,  -7.39846082e-06,  -8.00559920e-11],
      [  1.03267422e+00,  -1.10283816e-05,   5.30280579e-09,  -1.18715846e-12],
      [ -2.60199694e-05,  -3.04570646e-09,   1.12558669e-12,   1.40993647e-15],
      [  8.14712290e-09,   9.36542070e-13,  -4.20847687e-16,  -3.46570596e-19]]
Ky = [[  1.43440109e+01,   9.90752231e-01,  -3.52171557e-06,   7.17391873e-09],
      [ -2.43926351e-02,  -1.76691374e-05,   5.69247088e-09,  -2.86064608e-12],
      [  1.06635297e-05,   8.63408955e-09,  -2.66504801e-12,   1.47775242e-15],
      [ -1.10183664e-10,  -1.67574602e-13,   2.66154718e-16,  -1.13635710e-19]]
'''
# the below were taken in 2017B (with both apertures)
Kx = [[ -1.28092986e+01,   2.37843229e-02,  -8.09328998e-06,  -1.28301567e-10],
      [  1.02650326e+00,  -2.15600674e-05,   1.42563655e-08,  -3.38050198e-12],
      [ -1.94200903e-05,   3.51059572e-09,  -4.80126009e-12,   2.96495549e-15],
      [  6.26453161e-09,  -4.65825179e-13,   9.00810316e-16,  -7.28975474e-19]]
Ky = [[  1.47852158e+01,   9.92480569e-01,  -6.05953133e-06,   8.04550607e-09],
      [ -3.22153904e-02,  -1.77238376e-05,   1.09565607e-08,  -5.03884071e-12],
      [  1.90672398e-05,   3.96777274e-09,  -2.84884006e-12,   1.98359721e-15],
      [ -2.57086429e-09,   1.87084793e-12,  -3.96206006e-16,  -7.81470678e-20]]

#####################################################################
# DEWARP TEST FILE

hdul = fits.open('pinhole_image_median_DXonly_171108.fits') # this is just to get the shape
imagePinholes = hdul[0].data.copy()

# map the coordinates that define the entire image plane
dewarp_coords = dewarp.make_dewarp_coordinates(imagePinholes.shape,
                                               np.array(Kx).T,
                                               np.array(Ky).T) # transposed due to a coefficient definition change btwn Python and IDL


# grab the pre-dewarp image and header
imageAsterism, header = fits.getdata('pinhole_image_median_bothApertures_171108.fits',
                                     0, header=True)

# dewarp the image
dewarpedAsterism = dewarp.dewarp_with_precomputed_coords(imageAsterism,
                                                         dewarp_coords,
                                                         order=3)

# write out
fits.writeto('test_dewarped1.fits',
             np.squeeze(dewarpedAsterism),
             header,
             clobber=False)

#####################################################################
