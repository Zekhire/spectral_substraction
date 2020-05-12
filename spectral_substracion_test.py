import unittest
import numpy as np
import spectral_substracion

#TODO add tests for complex numbers and for float numbers!

class Power_spectral_density_estimation_of_the_noise(unittest.TestCase):
    def test_values(self):
        pass
    def test_len(self):
        pass



class Power_spectral_density_estimation_of_the_noisy_signal(unittest.TestCase):
    def test_values(self):
        # arrange
        N = 4
        Yi = np.array([1, -2, 3, 4])
        SYi_true = np.array([0.25, 1, 2.25])

        # act
        SYi = spectral_substracion.power_spectral_density_estimation_of_the_noisy_signal(Yi, N)

        # assert
        self.assertListEqual(list(SYi_true), list(SYi))


    def test_len(self):
        # arrange
        N=4
        Yi = np.array([1,2,3,4])
        NYi_true = 3

        # act
        NYi = len(spectral_substracion.power_spectral_density_estimation_of_the_noisy_signal(Yi, N))

        # assert
        self.assertEqual(NYi_true, NYi)



class Power_spectral_density_function_of_the_noiseless_signal(unittest.TestCase):
    def test_values(self):
        # arrange
        SYi = np.array([5, 6, 2, 10])
        SZ = np.array([1, 2, 2, 5])
        SXi_true = np.array([4, 4, 0, 5])

        # act
        SXi = spectral_substracion.power_spectral_density_function_of_the_noiseless_signal(SYi, SZ)

        #assert
        self.assertListEqual(list(SXi_true), list(SXi))
 
 
    def test_len(self):
        # arrange
        SYi = np.array([1, 2, 3, 4])
        SZ = np.array([4, 3, 2, 1])
        NXi_true = 4
        
        # act
        NXi = len(spectral_substracion.power_spectral_density_function_of_the_noiseless_signal(SYi, SZ))

        # assert
        self.assertEqual(NXi_true, NXi)



class Create_denoising_filter(unittest.TestCase):
    def test_values(self):
        # arrange
        SXi = np.array([72, 64, 16, 50])
        SYi = np.array([2, 4, 4, 2])
        Ai_true = np.array([6, 4, 2, 5, 2, 4])

        # act
        Ai = spectral_substracion.create_denoising_filter(SXi, SYi)

        # assert
        self.assertListEqual(list(Ai_true), list(Ai))


    def test_len(self):
        # arrange
        SXi = np.array([72, 64, 16, 50])
        SYi = np.array([2, 4, 4, 2])
        NAi_true = 6

        # act
        NAi = len(spectral_substracion.create_denoising_filter(SXi, SYi))

        # assert
        self.assertEqual(NAi_true, NAi)




class Evaluate_denoised_signal(unittest.TestCase):
    def test_values(self):
        # arrange
        Ai = np.array([1,2,9,4])
        Yi = np.array([5,3,2,3])
        Xi_true = np.array([5,6,18,12])

        # act
        Xi = spectral_substracion.evaluate_denoised_signal(Ai, Yi)

        # assert
        self.assertListEqual(list(Xi_true), list(Xi))


    def test_len(self):
        # arrange
        Ai = np.array([1,2,9,4])
        Yi = np.array([5,3,2,3])
        Xi_true = 4

        # act
        Xi = len(spectral_substracion.evaluate_denoised_signal(Ai, Yi))

        # assert
        self.assertEqual(Xi_true, Xi)