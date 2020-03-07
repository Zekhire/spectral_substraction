import unittest
import numpy as np
import spectral_substracion


class Test_power_spectral_density_estimation_of_the_noise(unittest.TestCase):
    def test_values(self):
        pass
    def test_len(self):
        pass




class Test_power_spectral_density_estimation_of_the_noisy_signal(unittest.TestCase):
    def test_values(self):
        N=4
        Yi = np.array([1,-2,3,4])
        SYi = np.array([0.25,1])
        self.assertListEqual(list(spectral_substracion.power_spectral_density_estimation_of_the_noisy_signal(Yi, N)), list(SYi))

        Yi = np.array([-16,-2,32223123,42111])
        SYi = np.array([64,1])
        self.assertListEqual(list(spectral_substracion.power_spectral_density_estimation_of_the_noisy_signal(Yi, N)), list(SYi))



    def test_len(self):
        N=4
        Yi = np.array([1,2,3,4])
        self.assertEqual(len(spectral_substracion.power_spectral_density_estimation_of_the_noisy_signal(Yi, N)), 2)


    


class Test_power_spectral_density_function_of_the_noiseless_signal(unittest.TestCase):
    def test_values(self):
        SYi = np.array([5,6,2,10])
        SZ = np.array([1,2,2,5])
        SXi = np.array([4,4,0,5])
        self.assertListEqual(list(spectral_substracion.power_spectral_density_function_of_the_noiseless_signal(SYi, SZ)), list(SXi))
        
        SYi = np.array([5,6,20,10])
        SZ = np.array([10,2,18,50])
        SXi = np.array([0,4,2,0])
        self.assertListEqual(list(spectral_substracion.power_spectral_density_function_of_the_noiseless_signal(SYi, SZ)), list(SXi))

    def test_len(self):
        SYi = np.array([1,2,3,4])
        SZ = np.array([4,3,2,1])
        self.assertEqual(len(spectral_substracion.power_spectral_density_function_of_the_noiseless_signal(SYi, SZ)), 4)



class Test_create_denoising_filter(unittest.TestCase):
    def test_values(self):
        SXi = np.array([72,64,16,50])
        SYi = np.array([2,4,4,2])
        Ai = np.array([np.sqrt(36), np.sqrt(16), np.sqrt(4), np.sqrt(25),
                        np.sqrt(25), np.sqrt(4), np.sqrt(16), np.sqrt(36)])

        self.assertListEqual(list(spectral_substracion.create_denoising_filter(SXi, SYi)), list(Ai))

    def test_len(self):
        SXi = np.array([72,64,16,50])
        SYi = np.array([2,4,4,2])
        self.assertEqual(len(spectral_substracion.create_denoising_filter(SXi, SYi)), 8)




class Test_evaluate_denoised_signal(unittest.TestCase):
    def test_values(self):
        Ai = np.array([1,2,9,4])
        Yi = np.array([5,3,2,3])
        Xi = np.array([5,6,18,12])

        self.assertListEqual(list(spectral_substracion.evaluate_denoised_signal(Ai, Yi)),list(Xi))

    def test_len(self):
        Ai = np.array([1,2,9,4])
        Yi = np.array([5,3,2,3])

        self.assertEqual(len(spectral_substracion.evaluate_denoised_signal(Ai, Yi)), 4)