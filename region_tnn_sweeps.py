# region_tnn_actor_critic_sweeps.py - W&B Sweeps Ready
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque
import gudhi
import toponetx as tnx
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import open3d as o3d
import wandb
import tempfile
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from torch.distributions import Categorical, Normal
from topomodelx.nn.simplicial.scconv import SCConv
from topomodelx.utils.sparse import from_sparse
import os


def capture_pcd_image(pcd_list, save_path=None):
    """
    Renders Open3D point clouds (pcd_list) to a static matplotlib image.
    Returns a numpy RGB array for wandb logging.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    for pcd in pcd_list:
        vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()

    if save_path is None:
        tmpfile = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        save_path = tmpfile.name

    vis.capture_screen_image(save_path)
    vis.destroy_window()

    img = plt.imread(save_path)
    return img, save_path


class RewardFunction:
    """Fixed reward function with proper magnitude checking."""
    def __init__(self, threshold=0.05, solved_bonus=100.0, move_penalty=-10.0):
        self.threshold = threshold
        self.solved_bonus = solved_bonus
        self.move_penalty = move_penalty

    def region_overlap_reward(self, pred_points, gt_points, magnitude=None):
        """
        Compute *per-point* reward mask for a region.

        Returns:
            reward_mask: np.array(len(pred_points)) → 1 if matched, else 0
            matched_gt: np.array(len(gt_points)) → True if GT point matched
            region_bonus: scalar reward for solved region (+100, +penalty)
            done: bool, whether all gt_points matched
        """
        if len(pred_points) == 0 or len(gt_points) == 0:
            return (
                np.zeros(len(pred_points), dtype=float),
                np.zeros(len(gt_points), dtype=bool),
                0.0,
                False
            )

        gt_tree = cKDTree(gt_points)
        matched_gt = np.zeros(len(gt_points), dtype=bool)
        reward_mask = np.zeros(len(pred_points), dtype=float)

        # For each predicted point, find nearest unmatched GT point
        for i, p in enumerate(pred_points):
            idxs = gt_tree.query_ball_point(p, self.threshold)
            available = [j for j in idxs if not matched_gt[j]]
            if len(available) > 0:
                matched_gt[available[0]] = True
                reward_mask[i] = 1.0

        # region completion check - FIXED
        all_matched = matched_gt.all()
        region_bonus = 0.0
        if all_matched:
            if magnitude is not None:
                mag_val = magnitude.item() if isinstance(magnitude, torch.Tensor) else magnitude
                if -0.05 <= mag_val <= 0.05:
                    region_bonus += self.solved_bonus
                else:
                    region_bonus += self.move_penalty

        return reward_mask, matched_gt, region_bonus, all_matched


def estimate_normals_pca(coords, k=16):
    """
    coords: (N, 3) torch tensor
    returns: (N, 3) unit normals estimated via local PCA (torch)
    """
    coords = coords.float()
    N = coords.shape[0]
    k = min(k, N-1)
    dists = torch.cdist(coords, coords)
    knn_idx = dists.topk(k=k+1, largest=False).indices[:, 1:]

    normals = torch.zeros_like(coords)
    for i in range(N):
        nbrs = coords[knn_idx[i]]
        center = coords[i].unsqueeze(0)
        cov = (nbrs - center).T @ (nbrs - center)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        normals[i] = eigvecs[:, 0]

    normals = normals / (torch.norm(normals, dim=1, keepdim=True) + 1e-8)
    return normals


def quasi_geodesic_distance(coords, normals):
    """
    coords: torch tensor (N,3)
    normals: torch tensor (N,3)
    returns: torch tensor (N,N) approximate distance
    """
    euclid = torch.cdist(coords, coords)
    dot = torch.matmul(normals, normals.T)
    return euclid * (2 - dot)


def generate_random_point_cloud(num_points=300, seed=42):
    np.random.seed(seed)
    return np.random.rand(num_points, 3).astype(np.float32)


def build_knn_sparse_matrix(points, max_distance=0.2):
    tree = cKDTree(points)
    coo = tree.sparse_distance_matrix(tree, max_distance=max_distance, output_type='coo_matrix')
    return coo


def build_simplex_tree_from_coo(points, coo, max_dim=2):
    st = gudhi.SimplexTree()
    n = points.shape[0]
    for i in range(n):
        st.insert([i])
    rows = coo.row
    cols = coo.col
    for i, j in zip(rows, cols):
        if i < j:
            st.insert([int(i), int(j)])
    st.expansion(max_dim)
    return st


def simplex_tree_to_toponetx(st):
    sc = tnx.SimplicialComplex()
    for simplex, _ in st.get_simplices():
        sc.add_simplex(list(simplex))
    return sc


def build_vr_complex_gpu(points, max_dim=4, epsilon=0.2):
    """
    Builds a Vietoris–Rips Complex using GUDHI on GPU tensors.
    """
    device = points.device
    points_cpu = points.cpu().numpy()

    rips_complex = gudhi.RipsComplex(points=points_cpu, max_edge_length=epsilon)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dim)

    return simplex_tree


class TNNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, n_layers=1):
        super().__init__()
        self.base_model = SCConv(node_channels=in_channels, n_layers=n_layers)
        self.linear_x0 = torch.nn.Linear(in_channels, out_channels)
        self.linear_x1 = torch.nn.Linear(in_channels, out_channels)
        self.linear_x2 = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x_0, x_1, x_2,
                incidence_1, incidence_1_norm,
                incidence_2, incidence_2_norm,
                adjacency_up_0_norm, adjacency_up_1_norm,
                adjacency_down_1_norm, adjacency_down_2_norm):
        x_0, x_1, x_2 = self.base_model(
            x_0, x_1, x_2,
            incidence_1, incidence_1_norm,
            incidence_2, incidence_2_norm,
            adjacency_up_0_norm, adjacency_up_1_norm,
            adjacency_down_1_norm, adjacency_down_2_norm,
        )
        x_0 = self.linear_x0(x_0)
        x_1 = self.linear_x1(x_1)
        x_2 = self.linear_x2(x_2)
        return x_0, x_1, x_2


class SharedTNNActorCritic(nn.Module):
    """Fixed Actor-Critic with proper architecture."""
    def __init__(self, in_channels=3, emb_dim=32, hidden_dim=128, n_directions=200):
        super().__init__()
        self.tnn = TNNEncoder(in_channels=in_channels, out_channels=emb_dim, n_layers=2)

        # Actor head
        self.actor_fc1 = nn.Linear(emb_dim, hidden_dim)
        self.actor_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dir_logits = nn.Linear(hidden_dim, n_directions)
        self.mag_mean = nn.Linear(hidden_dim, 1)
        self.mag_logstd = nn.Parameter(torch.zeros(1))

        # Critic head
        self.critic_fc1 = nn.Linear(emb_dim, hidden_dim)
        self.critic_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def get_region_embeddings(self, x0_emb, masks_ip):
        """Aggregate node embeddings by region."""
        region_embeddings = []
        for mask in masks_ip:
            idxs = np.where(mask)[0]
            if len(idxs) > 0:
                region_embeddings.append(x0_emb[idxs].mean(0))
            else:
                region_embeddings.append(torch.zeros(x0_emb.shape[1], device=x0_emb.device))
        return torch.stack(region_embeddings)

    def encode_state(self, x0, x1, x2,
                     incidence_1, incidence_1_norm,
                     incidence_2, incidence_2_norm,
                     adjacency_up_0_norm, adjacency_up_1_norm,
                     adjacency_down_1_norm, adjacency_down_2_norm,
                     masks_ip):
        """Encode full state → region embeddings."""
        x0_emb, x1_emb, x2_emb = self.tnn(
            x0, x1, x2,
            incidence_1, incidence_1_norm,
            incidence_2, incidence_2_norm,
            adjacency_up_0_norm, adjacency_up_1_norm,
            adjacency_down_1_norm, adjacency_down_2_norm
        )
        region_embeddings = self.get_region_embeddings(x0_emb, masks_ip)
        return region_embeddings

    def act(self, x0, x1, x2, inc1, inc1n, inc2, inc2n,
            adj_up0, adj_up1, adj_down1, adj_down2,
            masks_ip, DIRECTIONS, max_distance):
        """Sample actions from policy."""
        region_embeddings = self.encode_state(
            x0, x1, x2, inc1, inc1n, inc2, inc2n,
            adj_up0, adj_up1, adj_down1, adj_down2, masks_ip
        )

        x = F.relu(self.actor_fc1(region_embeddings))
        x = F.relu(self.actor_fc2(x))

        dir_dist = Categorical(logits=self.dir_logits(x))
        mag_dist = Normal(torch.tanh(self.mag_mean(x)), self.mag_logstd.exp())

        dir_idx = dir_dist.sample()
        magnitude = mag_dist.sample().clamp(-1, 1) * max_distance

        log_prob = (dir_dist.log_prob(dir_idx) + mag_dist.log_prob(magnitude.squeeze(-1)))
        sampled_directions = DIRECTIONS[dir_idx.cpu().numpy()]

        return (
            torch.tensor(sampled_directions, dtype=torch.float32, device=x.device),
            magnitude,
            log_prob,
            region_embeddings
        )

    def evaluate(self, region_embeddings):
        """Return V(s) for each region."""
        x = F.relu(self.critic_fc1(region_embeddings))
        x = F.relu(self.critic_fc2(x))
        v_value = self.value_head(x)
        return v_value


def select_region_centers(points, R=8, method='bbox_corners'):
    """Get extreme corners of point cloud as centres."""
    pts = np.asarray(points)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    bbox_corners = np.array([
        [mins[0], mins[1], mins[2]],
        [mins[0], mins[1], maxs[2]],
        [mins[0], maxs[1], mins[2]],
        [mins[0], maxs[1], maxs[2]],
        [maxs[0], mins[1], mins[2]],
        [maxs[0], mins[1], maxs[2]],
        [maxs[0], maxs[1], mins[2]],
        [maxs[0], maxs[1], maxs[2]],
    ])
    if method == 'random':
        choices = pts[np.random.choice(len(pts), R, replace=False)]
        return choices
    else:
        repeats = int(np.ceil(R/8))
        corners = np.tile(bbox_corners, (repeats,1))[:R]
        return corners


def build_region_masks(points, D, center_indices, K):
    """Assign each point to its single closest region center."""
    N = len(points)
    R = len(center_indices)
    masks = [np.zeros(N, dtype=bool) for _ in range(R)]

    dist_matrix = np.stack([D[ci] for ci in center_indices], axis=0)
    nearest_region = np.argmin(dist_matrix, axis=0)

    for r in range(R):
        masks[r][nearest_region == r] = True

    return masks


def reward_model(X_pred, X_gt, masks1=None, masks2=None, magnitudes=None):
    """
    Compute total scalar reward and done flag across all regions.
    """
    reward_fn = RewardFunction(threshold=0.05, solved_bonus=100.0, move_penalty=-10.0)

    if masks1 is None or masks2 is None:
        raise ValueError("Both masks1 and masks2 must be provided.")

    if len(masks1) != len(masks2):
        raise ValueError(f"Mismatch in region count: {len(masks1)} vs {len(masks2)}")

    R = len(masks1)
    total_reward_masks = []
    region_bonuses = []
    all_matched_global = []

    for r in range(R):
        mask_pred = masks1[r]
        mask_gt = masks2[r]

        gt_region = X_gt[mask_gt]
        pred_region = X_pred[mask_pred]

        magnitude_r = magnitudes[r] if magnitudes is not None else None

        reward_mask_r, matched_gt_r, region_bonus_r, all_matched_r = \
            reward_fn.region_overlap_reward(pred_region, gt_region, magnitude=magnitude_r)

        total_reward_masks.append(reward_mask_r)
        region_bonuses.append(region_bonus_r)
        all_matched_global.append(all_matched_r)

    reward_points_total = sum(mask.sum() for mask in total_reward_masks)
    reward_bonus_total = sum(region_bonuses)
    r_t = reward_points_total + reward_bonus_total

    return float(r_t), reward_bonus_total, total_reward_masks, all_matched_global


movement_history = {}


def apply_region_actions_with_rewards(
    X_pred, X_gt, pred_masks, gt_masks, reward_masks, vectors,
    all_matched_global, magnitudes=None, epoch=None, t_step=None
):
    """Apply per-region action vectors based on predicted reward masks."""
    global movement_history
    if magnitudes is None:
        magnitudes = [0.0] * len(pred_masks)

    with torch.no_grad():
        X_new = X_pred.clone()
        device = X_pred.device
        R = len(pred_masks)
        done = False
        moved_counts = []

        if all(all_matched_global):
            done = True
        else:
            update = torch.zeros_like(X_pred)

            for r in range(R):
                mask_pred_np = pred_masks[r]
                mask_gt_np = gt_masks[r]
                reward_mask_np = reward_masks[r]

                if mask_pred_np.sum() == 0 or mask_gt_np.sum() == 0:
                    moved_counts.append(0)
                    continue

                if reward_mask_np.shape[0] != mask_pred_np.sum():
                    print(f"[Error] Region {r}: reward_mask size mismatch with predicted region.")
                    moved_counts.append(0)
                    continue

                mask_pred = torch.tensor(mask_pred_np, dtype=torch.bool, device=device)
                reward_mask = torch.tensor(reward_mask_np, dtype=torch.bool, device=device)
                not_matched_mask = ~reward_mask

                if torch.any(not_matched_mask):
                    region_indices = mask_pred.nonzero(as_tuple=True)[0]
                    move_indices = region_indices[not_matched_mask]
                    moved_counts.append(move_indices.numel())

                    update[move_indices] = vectors[r].unsqueeze(0)
                else:
                    moved_counts.append(0)

            X_new = X_new + update

        log_dict = {"epoch": epoch, "timestep": t_step}
        for r in range(R):
            moved = moved_counts[r] if r < len(moved_counts) else 0
            mag = magnitudes[r].item() if isinstance(magnitudes[r], torch.Tensor) else magnitudes[r]
            log_dict[f"points_moved_region_{r}"] = moved
            log_dict[f"magnitude_region_{r}"] = mag

            if r not in movement_history:
                movement_history[r] = []
            movement_history[r].append(moved)

        log_dict["total_points_moved"] = sum(moved_counts)
        wandb.log(log_dict)

    return X_new, done,moved


def plot_colored_pointcloud(points, masks, title="Predicted", gt_points=None, gt_masks=None):
    """Visualize point cloud with regional colors."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    
    fig = plt.figure(figsize=(7, 6))
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111, projection='3d')
    cmap = plt.colormaps.get_cmap("tab20")

    for r, mask in enumerate(masks):
        pts = points[mask]
        if len(pts) == 0:
            continue
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   s=6, color=cmap(r % 20), label=f"Region {r}", alpha=0.6)

    if gt_points is not None and gt_masks is not None:
        for r, mask in enumerate(gt_masks):
            pts_gt = gt_points[mask]
            if len(pts_gt) == 0:
                continue
            ax.scatter(pts_gt[:, 0], pts_gt[:, 1], pts_gt[:, 2],
                       s=10, color=cmap(r % 20), marker="^", alpha=0.4, label=f"GT {r}")

    ax.set_title(title)
    ax.set_box_aspect([1, 1, 1])
    ax.legend(fontsize=6, loc='upper right', markerscale=2)
    ax.set_axis_off()

    canvas.draw()
    img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img = img[:, :, :3]
    plt.close(fig)
    return img


def match_region_masks_to_random_points(points, gt_masks):
    """Create region masks for random point cloud matching GT region sizes."""
    N = len(points)
    R = len(gt_masks)

    input_masks = [np.zeros(N, dtype=bool) for _ in range(R)]

    shuffled_indices = np.arange(N)
    np.random.shuffle(shuffled_indices)

    current = 0
    for r in range(R):
        num_points_region = int(np.sum(gt_masks[r]))
        if num_points_region <= 0:
            continue

        num_points_region = min(num_points_region, N - current)
        if num_points_region <= 0:
            break

        selected_indices = shuffled_indices[current:current + num_points_region]
        input_masks[r][selected_indices] = True
        current += num_points_region

    if current < N:
        leftover_indices = shuffled_indices[current:]
        region_sizes = [mask.sum() for mask in input_masks]
        for idx in leftover_indices:
            smallest = np.argmin(region_sizes)
            input_masks[smallest][idx] = True
            region_sizes[smallest] += 1

    return input_masks


def normalize_point_cloud(points):
    """Centers and scales a point cloud to fit inside a unit sphere."""
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    max_dist = np.max(np.linalg.norm(points_centered, axis=1))
    points_normalized = points_centered / max_dist
    return points_normalized


def load_ply_point_cloud(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    return np.asarray(pcd.points)


# Generate uniformly distributed directions on sphere
def generate_directions(n=200):
    """Generate n uniformly distributed directions on unit sphere."""
    phi = np.random.uniform(0, 2 * np.pi, n)
    cos_theta = np.random.uniform(-1, 1, n)
    theta = np.arccos(cos_theta)
    
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    return np.stack([x, y, z], axis=1).astype(np.float32)


def extract_features_from_complex(X_t, device):
    """
    Extract node, edge, face features and topological matrices from point cloud.
    Returns all tensors needed for TNN forward pass.
    """
    st = build_vr_complex_gpu(X_t, max_dim=4, epsilon=0.1)
    sc = simplex_tree_to_toponetx(st)
    X_np = X_t.detach().cpu().numpy().astype(np.float32)

    # Node features
    node_attr = {i: X_np[i] for i in range(X_np.shape[0])}
    sc.set_simplex_attributes(node_attr, name="node_feat")

    # Split simplices by dimension
    simplices = list(sc.simplices)
    simplices_2 = [s for s in simplices if len(s) == 2]
    simplices_3 = [s for s in simplices if len(s) == 3]

    # Edge features
    if simplices_2:
        edges = np.array(simplices_2, dtype=int)
        edge_feats = X_np[edges[:, 0]] - X_np[edges[:, 1]]
        edge_feats = edge_feats.astype(np.float32)
        for simplex, feat in zip(simplices_2, edge_feats):
            sc[simplex]["edge_feat"] = feat

    # Face features
    if simplices_3:
        faces = np.array(simplices_3, dtype=int)
        p1, p2, p3 = X_np[faces[:, 0]], X_np[faces[:, 1]], X_np[faces[:, 2]]
        normals = np.cross(p2 - p1, p3 - p1)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.divide(normals, norms, out=np.zeros_like(normals), where=norms != 0)
        face_feats = np.hstack([
            np.zeros((len(normals), 2), dtype=np.float32),
            normals.astype(np.float32)
        ])
        for simplex, feat in zip(simplices_3, face_feats):
            sc[simplex]["face_feat"] = feat

    # Collect into tensors
    x0_dict = sc.get_simplex_attributes("node_feat")
    x0 = torch.tensor(np.stack(list(x0_dict.values())), dtype=torch.float32, device=device)

    edge_attr_dict = sc.get_simplex_attributes("edge_feat")
    if edge_attr_dict:
        x1 = torch.tensor(np.stack(list(edge_attr_dict.values())), dtype=torch.float32, device=device)
    else:
        x1 = torch.zeros((0, 3), dtype=torch.float32, device=device)

    face_attr_dict = sc.get_simplex_attributes("face_feat")
    if face_attr_dict:
        x2 = torch.tensor(np.stack(list(face_attr_dict.values())), dtype=torch.float32, device=device)
        if x2.shape[1] != 3:
            proj = torch.nn.Linear(x2.shape[1], 3).to(device)
            x2 = proj(x2)
    else:
        x2 = torch.zeros((0, 3), dtype=torch.float32, device=device)

    # Convert topomodelx sparse matrices to torch sparse
    try:
        incidence_1 = from_sparse(sc.coincidence_matrix(1)).to(device)
        incidence_2 = from_sparse(sc.coincidence_matrix(2)).to(device)
        incidence_1_norm = from_sparse(sc.incidence_matrix(1)).to(device)
        incidence_2_norm = from_sparse(sc.incidence_matrix(2)).to(device)
        adjacency_up_0_norm = from_sparse(sc.up_laplacian_matrix(0)).to(device)
        adjacency_up_1_norm = from_sparse(sc.up_laplacian_matrix(1)).to(device)
        adjacency_down_1_norm = from_sparse(sc.down_laplacian_matrix(1)).to(device)
        adjacency_down_2_norm = from_sparse(sc.down_laplacian_matrix(2)).to(device)
    except Exception as e:
        n_nodes = x0.shape[0]
        incidence_1 = incidence_2 = incidence_1_norm = incidence_2_norm = torch.sparse_coo_tensor(
            indices=torch.zeros((2,0), dtype=torch.int64), 
            values=torch.tensor([], dtype=torch.float32), 
            size=(n_nodes, n_nodes)
        ).to(device)
        adjacency_up_0_norm = adjacency_up_1_norm = adjacency_down_1_norm = adjacency_down_2_norm = incidence_1

    return (x0, x1, x2, incidence_1, incidence_1_norm, incidence_2, incidence_2_norm,
            adjacency_up_0_norm, adjacency_up_1_norm, adjacency_down_1_norm, adjacency_down_2_norm)

def check_constant_last_n(records, n=10):
    if len(records) < n:
        return False
    last_n = records[-n:]
    return all(r == last_n[0] for r in last_n)

def train_region_actor_critic(config, gt_np):
    """
    FIXED: Complete Actor-Critic training loop with proper value bootstrapping.
    Now uses hyperparameters from W&B config.
    """
    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Extract hyperparameters from W&B config
    learning_rate = config.learning_rate
    weight_decay = config.weight_decay
    gamma = config.gamma
    entropy_coef = config.entropy_coef
    critic_weight = config.critic_weight
    gradient_clip = config.gradient_clip
    
    emb_dim = config.emb_dim
    hidden_dim = config.hidden_dim
    n_directions = config.n_directions
    max_dist = config.max_distance
    
    R = config.num_regions
    Tmax = config.tmax
    epochs = config.epochs
    reward_threshold = config.reward_threshold
    solved_bonus = config.solved_bonus
    move_penalty = config.move_penalty

    # Generate directions based on n_directions config
    DIRECTIONS = generate_directions(n_directions)
    
    policy = SharedTNNActorCritic(
        in_channels=3, emb_dim=emb_dim, hidden_dim=hidden_dim, n_directions=n_directions
    ).to(device)

    # FIXED: Single optimizer for all parameters
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Precompute GT regions
    gt = torch.tensor(gt_np, dtype=torch.float32, device=device)
    centers_gt = select_region_centers(gt_np, R)
    center_indicesg = [
        np.argmin(np.linalg.norm(gt_np - c, axis=1)) for c in centers_gt
    ]
    D_gt = quasi_geodesic_distance(gt, estimate_normals_pca(gt))
    masks_gt = build_region_masks(gt_np, D_gt.cpu().numpy(), center_indicesg, R)

    # Log GT visualization once
    img_gt = plot_colored_pointcloud(gt_np, masks_gt, title="Ground Truth Regions")
    wandb.log({"ground_truth_regions": wandb.Image(img_gt)})

    # save_dir = r"predicted_ply/"
    save_dir = os.path.join("predicted_ply_cup", str(R))
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        ep_reward = 0
        print(f"Max steps-{Tmax}")
        # Initialize state
        # Initialize Noisy object
        # points = load_ply_point_cloud(r"noisy_sphere.ply")
        # points = normalize_point_cloud(gt_np)

        # Initialize random point cloud
        points = generate_random_point_cloud(num_points=1000, seed=epoch)
        points = normalize_point_cloud(points)

        # noise = np.random.normal(0, 0.01, point_cloud.shape)
        X_t = torch.tensor(points, dtype=torch.float32, device=device)
        
        masks_ip = match_region_masks_to_random_points(points, masks_gt)
        overlap_masks = [np.zeros_like(mask, dtype=bool) for mask in masks_ip]
        epoch_pt_moved=[]
        for t in range(Tmax):
            # === Extract features for current state ===
            (x0, x1, x2, incidence_1, incidence_1_norm, incidence_2, incidence_2_norm,
             adjacency_up_0_norm, adjacency_up_1_norm, adjacency_down_1_norm, 
             adjacency_down_2_norm) = extract_features_from_complex(X_t, device)

            # === Get action from policy ===
            direction, magnitude, log_prob, region_embeddings = policy.act(
                x0, x1, x2,
                incidence_1, incidence_1_norm,
                incidence_2, incidence_2_norm,
                adjacency_up_0_norm, adjacency_up_1_norm,
                adjacency_down_1_norm, adjacency_down_2_norm,
                masks_ip,
                DIRECTIONS,
                max_distance=max_dist
            )

            # === Get value for current state ===
            v_s = policy.evaluate(region_embeddings).mean()  # Average across regions

            # === Apply action to environment ===
            vectors = direction * magnitude
            X_tp1, done, pt_moved = apply_region_actions_with_rewards(
                X_t, gt, masks_ip, masks_gt, overlap_masks,
                vectors, [False] * R, magnitude,
                epoch=epoch, t_step=t
            )
            epoch_pt_moved.append(pt_moved)
            # === Compute reward ===
            r_t, reward_bonus, reward_masks, all_matched_global = reward_model(
                X_tp1.cpu().numpy(), gt.cpu().numpy(),
                masks1=masks_ip, masks2=masks_gt, magnitudes=magnitude
            )
            r_t_tensor = torch.tensor(r_t, dtype=torch.float32, device=device)

            # === CRITICAL FIX: Compute next state value ===
            if not done:
                # Extract features for next state X_tp1
                (x0_next, x1_next, x2_next, 
                 incidence_1_next, incidence_1_norm_next,
                 incidence_2_next, incidence_2_norm_next,
                 adjacency_up_0_norm_next, adjacency_up_1_norm_next,
                 adjacency_down_1_norm_next, adjacency_down_2_norm_next) = \
                    extract_features_from_complex(X_tp1, device)

                with torch.no_grad():
                    region_embeddings_next = policy.encode_state(
                        x0_next, x1_next, x2_next,
                        incidence_1_next, incidence_1_norm_next,
                        incidence_2_next, incidence_2_norm_next,
                        adjacency_up_0_norm_next, adjacency_up_1_norm_next,
                        adjacency_down_1_norm_next, adjacency_down_2_norm_next,
                        masks_ip
                    )
                    v_s_next = policy.evaluate(region_embeddings_next).mean()
            else:
                v_s_next = torch.tensor(0.0, device=device)

            # === Compute TD error ===
            td_target = r_t_tensor + gamma * v_s_next
            delta = td_target - v_s

            # === Compute losses ===
            critic_loss = delta.pow(2)

            # Add entropy bonus for exploration
            entropy = -(torch.exp(log_prob) * log_prob).mean()
            actor_loss = -(log_prob * delta.detach()).mean() - entropy_coef * entropy

            # === Combined update ===
            total_loss = actor_loss + critic_weight * critic_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=gradient_clip)
            optimizer.step()

            # === Logging ===
            wandb.log({
                "step":t,
                "reward_step": r_t,
                "critic_loss": critic_loss.item(),
                "actor_loss": actor_loss.item(),
                "entropy": entropy.item(),
                "td_error": delta.item(),
                "value_estimate": v_s.item(),
            })

            # if (t + 1) % 100 == 0:
            #     pcd = o3d.geometry.PointCloud()
            #     pcd.points = o3d.utility.Vector3dVector(X_tp1.cpu().numpy())
            #     ply_filename = os.path.join(save_dir, f"epoch_{epoch+1}_step_{t+1}.ply")
            #     o3d.io.write_point_cloud(ply_filename, pcd)
            #     print(f"Saved predicted point cloud at {ply_filename}")

            # === Update state ===
            overlap_masks = reward_masks
            X_t = X_tp1
            ep_reward += r_t

            indicator = check_constant_last_n(epoch_pt_moved, 10)
            if indicator:
                # print("Indicator: Last 10 rewards are constant \n")
                print(f"Episode {epoch+1} break at step {t+1}")
                break
            if done:
                print(f"Epoch {epoch+1}, Step {t+1}: Terminal state reached!")
                break
        # if (t + 1) % 100 == 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(X_tp1.cpu().numpy())
        ply_filename = os.path.join(save_dir, f"epoch_{epoch+1}_step_{t+1}.ply")
        o3d.io.write_point_cloud(ply_filename, pcd)
        print(f"Saved predicted point cloud at {ply_filename}")

        # === End of epoch logging ===
        wandb.log({
            "epoch": epoch + 1,
            "reward_epoch": ep_reward,
            "points_overlapped":pt_moved,
        })

        # Visualize final prediction every 10 epochs
        # if (epoch + 1) % 1 == 0:
        img_pred = plot_colored_pointcloud(
            X_t.cpu().numpy(),
            masks_ip,
            title=f"Epoch {epoch+1} Final Prediction"
        )
        wandb.log({f"pred_cloud/epoch{epoch+1}": wandb.Image(img_pred)})

        print(f"Epoch {epoch+1}/{epochs} - Total Reward: {ep_reward:.2f}")

    wandb.finish()
    return X_t.cpu().numpy(), policy


def train_agent():
    """W&B sweep compatible training function."""
    with wandb.init(config=None) as run:
        config = run.config
        
        # Load ground truth point cloud
        gt_np = load_ply_point_cloud("cup_0003_1000_normalised.ply")
        gt_np = normalize_point_cloud(gt_np)
        
        print(f"Loaded ground truth point cloud with {gt_np.shape[0]} points.")
        
        # Train the model
        X_hat, trained_policy = train_region_actor_critic(config, gt_np)
        
        print("Training complete!")
        print(f"Final prediction shape: {X_hat.shape}")


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    
    train_agent()
