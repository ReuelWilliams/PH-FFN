import numpy as np
import torch

class Manifold:

    def __init__(self, num_points: int, dim: int, sampling_function=None, label_function = None) -> None:
        self.num_points = num_points
        self.dim = dim
        self.sampling_function = sampling_function
        self.label_function = label_function

    def get_training_data(self):
        points = self.sampling_function(self.num_points, self.dim)
        labels = self.label_function(points)
        return points, labels
    def get_test_data(self):
        points = self.sampling_function(self.num_points, self.dim)
        labels = self.label_function(points)
        return points, labels
    
    def get_name(self):
        return f"{self.sampling_function.__name__}_{self.label_function.__name__}"




def sample_from_n_sphere(num_points : int, dim: int) -> np.ndarray:
    """Sample points from the n-sphere.

    Returns:
        np.ndarray: Array of shape (num_points, dim) containing the sampled points.
    """
    # Sample points from the standard normal distribution
    points = np.random.randn(num_points, dim)

    # Normalize the points to lie on the n-sphere
    points /= np.linalg.norm(points, axis=1)[:, None]

    # Convert points to a tensor
    points = torch.tensor(points, dtype=torch.float32)

    return points

def label_left_right(points: np.ndarray) -> np.ndarray:
    """Label points as left or right of the origin.

    Args:
        points (np.ndarray): Array of shape (num_points, dim) containing the points.
        n_classes (int): Number of classes. We need this for one-hot vector enconding.

    Returns:
        np.ndarray: Array of shape (num_points, n_classes) containing the labels as one hot vectors.
    """
    labels = np.where(points[:, 0] > 0, 1, 0)
    return torch.tensor(labels, dtype=torch.long)

def sample_from_russian_sphere(num_points : int, dim: int) -> np.ndarray:
    """Sample points from two n-sphere of radii r and r_prime.

    Returns:
        np.ndarray: Array of shape (num_points, dim) containing the sampled points.
    """
    
    # Sample points from the standard normal distribution
    points = np.random.randn(int(num_points / 2), dim)
    point_prime = np.random.randn(int(num_points / 2), dim)

    # Normalize the points to lie on the n-sphere
    points /= np.linalg.norm(points, axis=1)[:, None]

    point_prime /= np.linalg.norm(point_prime, axis=1)[:, None]
    point_prime *= 2 # outer sphere

    

    # Convert points to a tensor
    points = torch.tensor(points, dtype=torch.float32)
    point_prime = torch.tensor(point_prime, dtype=torch.float32)

    # Collapse points and point_prime along the first dimension
    points = torch.cat((points, point_prime), dim=0)

    indices = torch.randperm(points.shape[0])
    points = points[indices]

    return points

def label_inner_outer(points: np.ndarray) -> np.ndarray:
    """Label points as inner or outer of the origin.

    Args:
        points (np.ndarray): Array of shape (num_points, dim) containing the points.
        n_classes (int): Number of classes. We need this for one-hot vector enconding.

    Returns:
        np.ndarray: Array of shape (num_points, n_classes) containing the labels as one hot vectors.
    """
    labels = np.where(np.linalg.norm(points, axis=1) <= 1.5, 1, np.where(np.linalg.norm(points, axis=1) > 1.5 , 0, 3))
    return torch.tensor(labels, dtype=torch.long)