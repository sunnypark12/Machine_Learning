import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio


class PCA(object):

    def __init__(self):
        self.U = None
        self.S = None
        self.V = None

    def fit(self, X: np.ndarray) ->None:
        """		
		Decompose dataset into principal components by finding the singular value decomposition of the centered dataset X
		You may use the numpy.linalg.svd function
		Don't return anything. You can directly set self.U, self.S and self.V declared in __init__ with
		corresponding values from PCA. See the docstrings below for the expected shapes of U, S, and V transpose
		
		Hint: np.linalg.svd by default returns the transpose of V
		      Make sure you remember to first center your data by subtracting the mean of each feature.
		
		Args:
		    X: (N,D) numpy array corresponding to a dataset
		
		Return:
		    None
		
		Set:
		    self.U: (N, min(N,D)) numpy array
		    self.S: (min(N,D), ) numpy array
		    self.V: (min(N,D), D) numpy array
		"""
        self.U, self.S, self.V = np.linalg.svd(X - np.mean(X, axis=0), full_matrices=False)

    def transform(self, data: np.ndarray, K: int=2) ->np.ndarray:
        """		
		Transform data to reduce the number of features such that final data (X_new) has K features (columns)
		Utilize self.U, self.S and self.V that were set in fit() method.
		
		Args:
		    data: (N,D) numpy array corresponding to a dataset
		    K: int value for number of columns to be kept
		
		Return:
		    X_new: (N,K) numpy array corresponding to data obtained by applying PCA on data
		
		Hint: Make sure you remember to first center your data by subtracting the mean of each feature.
		"""

        return np.dot(data - np.mean(data, axis=0), self.V.T[:, :K])

    def transform_rv(self, data: np.ndarray, retained_variance: float=0.99
        ) ->np.ndarray:
        """		
		Transform data to reduce the number of features such that the retained variance given by retained_variance is kept
		in X_new with K features
		Utilize self.U, self.S and self.V that were set in fit() method.
		
		Args:
		    data: (N,D) numpy array corresponding to a dataset
		    retained_variance: float value for amount of variance to be retained
		
		Return:
		    X_new: (N,K) numpy array corresponding to data obtained by applying PCA on data, where K is the number of columns
		           to be kept to ensure retained variance value is retained_variance
		
		Hint: Make sure you remember to first center your data by subtracting the mean of each feature.
		"""
        return self.transform(data, np.searchsorted(np.cumsum(self.S ** 2) / np.sum(self.S ** 2), retained_variance) + 1)

    def get_V(self) ->np.ndarray:
        """		
		Getter function for value of V
		"""
        return self.V

    def visualize(self, X: np.ndarray, y: np.ndarray, fig_title) ->None:
        """		
		You have to plot three different scatterplots (2d and 3d for strongest 2 features and 2d for weakest 2 features) for this function. For plotting the 2d scatterplots, use your PCA implementation to reduce the dataset to only 2 (strongest and later weakest) features. You'll need to run PCA on the dataset and then transform it so that the new dataset only has 2 features.
		Create a scatter plot of the reduced data set and differentiate points that have different true labels using color using plotly.
		Hint: Refer to https://plotly.com/python/line-and-scatter/ for making scatter plots with plotly.
		Hint: We recommend converting the data into a pandas dataframe before plotting it. Refer to https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html for more details.
		Hint: To extract weakest features, consider the order of components returned in PCA.
		
		Args:
		    xtrain: (N,D) numpy array, where N is number of instances and D is the dimensionality of each instance
		    ytrain: (N,) numpy array, the true labels
		
		Return: None
		"""

        # Fit PCA
        self.fit(X)
        
        # Transform data for strongest 2 features
        X_strongest_2 = self.transform(X, 2)
        X_strongest_3 = self.transform(X, 3)

        # Transform data for weakest 2 features
        X_weakest_2 = np.dot(X - np.mean(X, axis=0), self.V.T[:, -2:])
        
        # Convert to DataFrame for plotting
        df_strongest_2 = pd.DataFrame(X_strongest_2, columns=['PC1', 'PC2'])
        df_strongest_2['label'] = y
        
        df_strongest_3 = pd.DataFrame(X_strongest_3, columns=['PC1', 'PC2', 'PC3'])
        df_strongest_3['label'] = y

        df_weakest_2 = pd.DataFrame(X_weakest_2, columns=['PC_weak_1', 'PC_weak_2'])
        df_weakest_2['label'] = y

        # 2D Scatter plot for strongest 2 features
        fig_strongest_2d = px.scatter(df_strongest_2, x='PC1', y='PC2', color='label', title=f'{fig_title} - Strongest 2 PCs')
        fig_strongest_2d.show(renderer='png')

        # 3D Scatter plot for strongest 3 features (if needed, adjust accordingly)
        fig_strongest_3d = px.scatter_3d(df_strongest_3, x='PC1', y='PC2', z='PC3', color='label', title=f'{fig_title} - Strongest 3 PCs')
        fig_strongest_3d.show(renderer='png')

        # 2D Scatter plot for weakest 2 features
        fig_weakest_2d = px.scatter(df_weakest_2, x='PC_weak_1', y='PC_weak_2', color='label', title=f'{fig_title} - Weakest 2 PCs')
        fig_weakest_2d.show(renderer='png')
        
