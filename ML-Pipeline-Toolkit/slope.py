import numpy as np
from pca import PCA
from regression import Regression


class Slope(object):

    def __init__(self):
        pass

    @staticmethod
    def pca_slope(X, y):
        """		
		Calculates the slope of the first principal component given by PCA
		
		Args:
		    x: N x 1 array of feature x
		    y: N x 1 array of feature y
		Return:
		    slope: (float) scalar slope of the first principal component
		"""
        # Stack the X and y vectors into an N x 2 matrix
        data = np.column_stack((X, y))
        
        # Perform PCA
        pca = PCA()
        pca.fit(data)
        
        # Get the first principal component
        first_pc = pca.V[0]
        
        # Calculate the slope
        slope = first_pc[1] / first_pc[0]
        return slope


    @staticmethod
    def lr_slope(X, y):
        """		
		Calculates the slope of the best fit returned by linear_fit_closed()
		
		For this function don't use any regularization
		
		Args:
		    X: N x 1 array corresponding to a dataset
		    y: N x 1 array of labels y
		Return:
		    slope: (float) slope of the best fit
		"""
        # Add a bias term to the feature matrix
        X_with_bias = np.column_stack((np.ones(X.shape[0]), X))
        
        # Perform linear regression
        reg = Regression()
        weights = reg.linear_fit_closed(X_with_bias, y)
        
        # The slope is the weight corresponding to the feature X (not the bias term)
        slope = weights[1][0]
        return slope

    @classmethod
    def addNoise(cls, c, x_noise=False, seed=1):
        """		
		Creates a dataset with noise and calculates the slope of the dataset
		using the pca_slope and lr_slope functions implemented in this class.
		
		Args:
		    c: (float) scalar, a given noise level to be used on Y and/or X
		    x_noise: (Boolean) When set to False, X should not have noise added
		            When set to True, X should have noise.
		            Note that the noise added to X should be different from the
		            noise added to Y. You should NOT use the same noise you add
		            to Y here.
		    seed: (int) Random seed
		Return:
		    pca_slope_value: (float) slope value of dataset created using pca_slope
		    lr_slope_value: (float) slope value of dataset created using lr_slope
		"""
        np.random.seed(seed)
        
        # Create dataset
        N = 1260  # Example size
        X = np.random.rand(N, 1)
        Y = 5 * X # Example linear relation
        
        # Add noise
        if x_noise:
            X += c * np.random.randn(N, 1)
        Y += c * np.random.randn(N, 1)
        
        # Calculate slopes
        pca_slope_value = cls.pca_slope(X, Y)
        lr_slope_value = cls.lr_slope(X, Y)
        
        return pca_slope_value, lr_slope_value