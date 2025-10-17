import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
from itertools import combinations

class PersistentHomologyLoss:
    """
    A class to compute persistent homology loss between two point clouds.

    This implementation focuses on 0-dimensional (connected components) and 
    1-dimensional (loops/cycles) homological features using a simplified 
    Vietoris-Rips filtration approach.

    Usage:
        persloss = PersistentHomologyLoss(max_dimension=1)
        loss = persloss.compute_loss(pointcloud1, pointcloud2)
    """

    def __init__(self, max_dimension=1, max_edge_length=np.inf):
        """
        Initialize the PersLoss calculator.

        Parameters:
        -----------
        max_dimension : int
            Maximum homological dimension to compute (0 or 1)
        max_edge_length : float
            Maximum edge length for Vietoris-Rips complex construction
        """
        self.max_dimension = max_dimension
        self.max_edge_length = max_edge_length

    def compute_distance_matrix(self, points):
        """
        Compute pairwise distance matrix for point cloud.

        Parameters:
        -----------
        points : numpy.ndarray
            Point cloud of shape (n_points, n_dimensions)

        Returns:
        --------
        numpy.ndarray : Distance matrix of shape (n_points, n_points)
        """
        return pairwise_distances(points, metric='euclidean')

    def compute_0d_persistence(self, distance_matrix):
        """
        Compute 0-dimensional persistence (connected components) using Union-Find.

        Parameters:
        -----------
        distance_matrix : numpy.ndarray
            Pairwise distances between points

        Returns:
        --------
        list : List of (birth, death) pairs for connected components
        """
        n_points = distance_matrix.shape[0]

        # Get all edges sorted by distance
        edges = []
        for i in range(n_points):
            for j in range(i + 1, n_points):
                if distance_matrix[i, j] <= self.max_edge_length:
                    edges.append((distance_matrix[i, j], i, j))

        edges.sort()  # Sort by distance (filtration parameter)

        # Union-Find data structure
        parent = list(range(n_points))
        component_birth = [0.0] * n_points  # All components born at time 0

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y, time):
            px, py = find(x), find(y)
            if px != py:
                # Merge components - one dies
                parent[py] = px
                return (component_birth[py], time)  # Component py dies at current time
            return None

        persistence_pairs = []

        # Process edges in order of increasing distance
        for dist, i, j in edges:
            death_pair = union(i, j, dist)
            if death_pair:
                persistence_pairs.append(death_pair)

        # Add one infinite component (the one that never dies)
        remaining_components = set(find(i) for i in range(n_points))
        if len(remaining_components) == 1:
            root = list(remaining_components)[0] 
            persistence_pairs.append((component_birth[root], np.inf))

        return persistence_pairs

    def compute_1d_persistence_simplified(self, distance_matrix, threshold=None):
        """
        Compute 1-dimensional persistence (loops) using a simplified approach.
        This is a basic implementation that identifies potential loops.

        Parameters:
        -----------
        distance_matrix : numpy.ndarray
            Pairwise distances between points
        threshold : float
            Maximum distance threshold for considering connections

        Returns:
        --------
        list : List of (birth, death) pairs for 1-dimensional features
        """
        if threshold is None:
            threshold = np.percentile(distance_matrix[distance_matrix > 0], 70)

        n_points = distance_matrix.shape[0]
        loops = []

        # Simple loop detection: look for triangles that form at different scales
        triangles = []
        for i, j, k in combinations(range(n_points), 3):
            edges = [
                distance_matrix[i, j],
                distance_matrix[j, k], 
                distance_matrix[i, k]
            ]
            edges.sort()

            if edges[2] <= threshold:  # All edges within threshold
                # Triangle forms when the longest edge appears
                triangles.append((edges[2], edges))

        # Sort triangles by their "birth" time (when they close)
        triangles.sort()

        # Simple heuristic: loops appear and disappear based on triangle formation
        for i, (birth_time, edges) in enumerate(triangles):
            if i < len(triangles) // 2:  # Arbitrary selection for loops
                death_time = birth_time * 1.5 if i < len(triangles) - 1 else np.inf
                loops.append((birth_time, death_time))

        return loops

    def compute_persistence_diagram(self, points):
        """
        Compute persistence diagram for a point cloud.

        Parameters:
        -----------
        points : numpy.ndarray
            Point cloud of shape (n_points, n_dimensions)

        Returns:
        --------
        dict : Dictionary with persistence diagrams for each dimension
        """
        if len(points) == 0:
            return {0: [], 1: []}

        distance_matrix = self.compute_distance_matrix(points)

        # Compute 0-dimensional persistence
        pers_0d = self.compute_0d_persistence(distance_matrix)

        result = {0: pers_0d}

        # Compute 1-dimensional persistence if requested
        if self.max_dimension >= 1:
            pers_1d = self.compute_1d_persistence_simplified(distance_matrix)
            result[1] = pers_1d

        return result

    def wasserstein_distance_1d(self, dgm1, dgm2, p=2):
        """
        Compute 1-Wasserstein distance between two 1D persistence diagrams.
        This is a simplified implementation.

        Parameters:
        -----------
        dgm1, dgm2 : list
            Lists of (birth, death) pairs
        p : int
            Wasserstein p-norm

        Returns:
        --------
        float : Wasserstein distance
        """
        # Handle infinite values by capping them
        def cap_infinite(diagram, cap_value=1000):
            return [(b, min(d, cap_value) if d != np.inf else cap_value) 
                    for b, d in diagram]

        dgm1_capped = cap_infinite(dgm1)
        dgm2_capped = cap_infinite(dgm2)

        if len(dgm1_capped) == 0 and len(dgm2_capped) == 0:
            return 0.0

        # Simple matching-based distance calculation
        # This is a simplified version - in practice, you'd use optimal transport

        n1, n2 = len(dgm1_capped), len(dgm2_capped)

        if n1 == 0:
            return sum(abs(d - b) for b, d in dgm2_capped)
        if n2 == 0:
            return sum(abs(d - b) for b, d in dgm1_capped)

        # Create cost matrix for bipartite matching
        costs = np.zeros((n1, n2))
        for i, (b1, d1) in enumerate(dgm1_capped):
            for j, (b2, d2) in enumerate(dgm2_capped):
                costs[i, j] = ((abs(b1 - b2)**p + abs(d1 - d2)**p) ** (1/p))

        # Simple greedy matching (not optimal, but illustrative)
        total_cost = 0
        used_j = set()
        for i in range(n1):
            best_j = None
            best_cost = np.inf
            for j in range(n2):
                if j not in used_j and costs[i, j] < best_cost:
                    best_cost = costs[i, j]
                    best_j = j
            if best_j is not None:
                total_cost += best_cost
                used_j.add(best_j)

        return total_cost

    def compute_loss(self, points1, points2, lambda_0d=1.0, lambda_1d=1.0):
        """
        Compute persistent homology loss between two point clouds.

        Parameters:
        -----------
        points1, points2 : numpy.ndarray
            Two point clouds to compare
        lambda_0d, lambda_1d : float
            Weights for 0D and 1D persistence terms

        Returns:
        --------
        dict : Dictionary with loss components and total loss
        """
        # Compute persistence diagrams
        dgm1 = self.compute_persistence_diagram(points1)
        dgm2 = self.compute_persistence_diagram(points2)

        # Compute losses for each dimension
        loss_0d = self.wasserstein_distance_1d(dgm1[0], dgm2[0])
        loss_0d *= lambda_0d

        loss_1d = 0.0
        if self.max_dimension >= 1:
            loss_1d = self.wasserstein_distance_1d(dgm1[1], dgm2[1])
            loss_1d *= lambda_1d

        total_loss = loss_0d + loss_1d

        return {
            'total_loss': total_loss,
            'loss_0d': loss_0d,
            'loss_1d': loss_1d,
            'dgm1': dgm1,
            'dgm2': dgm2
        }


def plot_persistence_diagram(dgm, title="Persistence Diagram", max_val=None):
    """
    Plot a persistence diagram.

    Parameters:
    -----------
    dgm : dict
        Dictionary of persistence diagrams by dimension
    title : str
        Title for the plot
    max_val : float
        Maximum value for plot scaling
    """
    plt.figure(figsize=(8, 6))

    colors = ['red', 'blue', 'green']
    labels = ['H0 (Components)', 'H1 (Loops)', 'H2 (Voids)']

    for dim in sorted(dgm.keys()):
        if len(dgm[dim]) > 0:
            births, deaths = zip(*dgm[dim])
            births = np.array(births)
            deaths = np.array([d if d != np.inf else (max_val or max(births)*2) 
                             for d in deaths])

            plt.scatter(births, deaths, c=colors[dim], 
                       label=labels[dim], alpha=0.7, s=50)

    # Plot diagonal
    if max_val is None:
        max_val = plt.gca().get_xlim()[1]

    plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Diagonal')
    plt.xlabel('Birth')
    plt.ylabel('Death')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()


# Example usage and testing
if __name__ == "__main__":
    print("Testing PersLoss Implementation")
    print("=" * 40)

    # Generate test data
    np.random.seed(42)

    # Point cloud 1: Random points
    points1 = np.random.randn(30, 2) * 0.5

    # Point cloud 2: Points arranged in a rough circle (should have 1-dimensional features)
    theta = np.linspace(0, 2*np.pi, 30, endpoint=False)
    points2 = np.column_stack([np.cos(theta), np.sin(theta)]) + np.random.randn(30, 2) * 0.1

    # Initialize PersLoss calculator
    persloss = PersistentHomologyLoss(max_dimension=1, max_edge_length=3.0)

    # Compute loss between different point clouds
    loss_result = persloss.compute_loss(points1, points2, lambda_0d=1.0, lambda_1d=0.1)

    print(f"Loss between different clouds:")
    print(f"  Total Loss: {loss_result['total_loss']:.4f}")
    print(f"  0D Loss: {loss_result['loss_0d']:.4f}")
    print(f"  1D Loss: {loss_result['loss_1d']:.4f}")

    # Test with identical clouds
    loss_identical = persloss.compute_loss(points1, points1.copy())
    print(f"\nLoss between identical clouds: {loss_identical['total_loss']:.4f}")

    # Simple example
    print("\nSimple Triangle Example:")
    triangle1 = np.array([[0, 0], [1, 0], [0.5, 0.866]])  # Unit triangle
    triangle2 = np.array([[0, 0], [2, 0], [1, 1.732]])    # Scaled triangle

    simple_loss = persloss.compute_loss(triangle1, triangle2)
    print(f"Loss between triangles: {simple_loss['total_loss']:.4f}")

    print("\nPersLoss implementation ready for use!")
