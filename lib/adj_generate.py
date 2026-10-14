import pickle
import numpy as np
from scipy.sparse.linalg import eigs
import scipy.sparse as sp
from scipy.sparse import linalg


def get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename=None):
    """
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    """
    if "npy" in distance_df_filename:

        adj_mx = np.load(distance_df_filename)

        return adj_mx, None

    else:

        import csv

        A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)

        distaneA = np.zeros(
            (int(num_of_vertices), int(num_of_vertices)), dtype=np.float32
        )
        distaneA[:] = np.inf

        # distance file中的id并不是从0开始的 所以要进行重新的映射；id_filename是节点的顺序
        if id_filename:
            with open(id_filename, "r") as f:
                id_dict = {
                    int(i): idx for idx, i in enumerate(f.read().strip().split("\n"))
                }  # 把节点id（idx）映射成从0开始的索引

            with open(distance_df_filename, "r") as f:
                f.readline()  # 略过表头那一行
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    distaneA[id_dict[i], id_dict[j]] = distance

        else:  # distance file中的id直接从0开始
            with open(distance_df_filename, "r") as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    distaneA[i, j] = distance

        distances = distaneA[~np.isinf(distaneA)].flatten()
        std = distances.std()
        distaneA = np.exp(-np.square(distaneA / std))
        distaneA[distaneA < 0.1] = 0.0
        return A, distaneA


def get_adjacency_matrix_2direction(
    distance_df_filename, num_of_vertices, id_filename=None
):
    """
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    """
    if "npy" in distance_df_filename:

        adj_mx = np.load(distance_df_filename)

        return adj_mx, None

    else:

        import csv

        A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)

        distaneA = np.zeros(
            (int(num_of_vertices), int(num_of_vertices)), dtype=np.float32
        )
        distaneA[:] = np.inf

        # distance file中的id并不是从0开始的 所以要进行重新的映射；id_filename是节点的顺序
        if id_filename:

            with open(id_filename, "r") as f:
                id_dict = {
                    int(i): idx for idx, i in enumerate(f.read().strip().split("\n"))
                }  # 把节点id（idx）映射成从0开始的索引

            with open(distance_df_filename, "r") as f:
                f.readline()  # 略过表头那一行
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    A[id_dict[j], id_dict[i]] = 1
                    distaneA[id_dict[i], id_dict[j]] = distance
                    distaneA[id_dict[j], id_dict[i]] = distance

        else:  # distance file中的id直接从0开始
            with open(distance_df_filename, "r") as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    A[j, i] = 1
                    distaneA[i, j] = distance
                    distaneA[j, i] = distance
        distances = distaneA[~np.isinf(distaneA)].flatten()
        std = distances.std()
        distaneA = np.exp(-np.square(distaneA / std))
        distaneA[distaneA < 0.1] = 0.0
        return A, distaneA


def calculate_symmetric_normalized_laplacian(adj: np.ndarray) -> np.matrix:
    """Calculate symmetric normalized laplacian.
    Assuming unnormalized laplacian matrix is `L = D - A`,
    then symmetric normalized laplacian matrix is:
    `L^{Sym} =  D^-1/2 L D^-1/2 =  D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2`
    For node `i` and `j` where `i!=j`, L^{sym}_{ij} <=0.

    Args:
        adj (np.ndarray): Adjacent matrix A

    Returns:
        np.matrix: Symmetric normalized laplacian L^{Sym}
    """

    adj = sp.coo_matrix(adj)
    degree = np.array(adj.sum(1))
    # diagonals of D^{-1/2}
    degree_inv_sqrt = np.power(degree, -0.5).flatten()
    degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.0
    matrix_degree_inv_sqrt = sp.diags(degree_inv_sqrt)  # D^{-1/2}
    symmetric_normalized_laplacian = (
        sp.eye(adj.shape[0])
        - matrix_degree_inv_sqrt.dot(adj).dot(matrix_degree_inv_sqrt).tocoo()
    )
    return symmetric_normalized_laplacian


def calculate_scaled_laplacian(
    adj: np.ndarray, lambda_max: int = 2, undirected: bool = True
) -> np.matrix:
    """Re-scaled the eigenvalue to [-1, 1] by scaled the normalized laplacian matrix for chebyshev pol.
    According to `2017 ICLR GCN`, the lambda max is set to 2, and the graph is set to undirected.
    Note that rescale the laplacian matrix is equal to rescale the eigenvalue matrix.
    `L_{scaled} = (2 / lambda_max * L) - I`

    Args:
        adj (np.ndarray): Adjacent matrix A
        lambda_max (int, optional): Defaults to 2.
        undirected (bool, optional): Defaults to True.

    Returns:
        np.matrix: The rescaled laplacian matrix.
    """

    if undirected:
        adj = np.maximum.reduce([adj, adj.T])
    laplacian_matrix = calculate_symmetric_normalized_laplacian(adj)
    if lambda_max is None:  # manually cal the max lambda
        lambda_max, _ = linalg.eigsh(laplacian_matrix, 1, which="LM")
        lambda_max = lambda_max[0]
    laplacian_matrix = sp.csr_matrix(laplacian_matrix)
    num_nodes, _ = laplacian_matrix.shape
    identity_matrix = sp.identity(num_nodes, format="csr", dtype=laplacian_matrix.dtype)
    laplacian_res = (2 / lambda_max * laplacian_matrix) - identity_matrix
    return laplacian_res


def calculate_symmetric_message_passing_adj(adj: np.ndarray) -> np.matrix:
    """Calculate the renormalized message passing adj in `GCN`.
    A = A + I
    return D^{-1/2} A D^{-1/2}

    Args:
        adj (np.ndarray): Adjacent matrix A

    Returns:
        np.matrix: Renormalized message passing adj in `GCN`.
    """

    # add self loop
    adj = adj + np.diag(np.ones(adj.shape[0], dtype=np.float32))
    # print("calculating the renormalized message passing adj, please ensure that self-loop has added to adj.")
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    mp_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).astype(np.float32)
    return mp_adj


def calculate_transition_matrix(adj: np.ndarray) -> np.matrix:
    """Calculate the transition matrix `P` proposed in DCRNN and Graph WaveNet.
    P = D^{-1}A = A/rowsum(A)

    Args:
        adj (np.ndarray): Adjacent matrix A

    Returns:
        np.matrix: Transition matrix P
    """
    # adj = adj + np.diag(np.ones(adj.shape[0], dtype=np.float32))

    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(row_sum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.0
    d_mat = sp.diags(d_inv)
    prob_matrix = d_mat.dot(adj).astype(np.float32)
    return prob_matrix


def load_pkl(pickle_file: str) -> object:
    """Load pickle data.

    Args:
        pickle_file (str): file path

    Returns:
        object: loaded objected
    """

    try:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f, encoding="latin1")
    except Exception as e:
        print("Unable to load data ", pickle_file, ":", e)
        raise
    return pickle_data


def load_adj(file_path: str, adj_type: str = "doubletransition"):
    """load adjacency matrix.

    Args:
        file_path (str): file path
        adj_type (str): adjacency matrix type

    Returns:
        list of numpy.matrix: list of preproceesed adjacency matrices
        np.ndarray: raw adjacency matrix
    """

    try:
        # METR and PEMS_BAY
        _, _, adj_mx = load_pkl(file_path)
    except ValueError:
        # PEMS04
        adj_mx = load_pkl(file_path)
    if adj_type == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx).astype(np.float32).todense()]
    elif adj_type == "normlap":
        adj = [
            calculate_symmetric_normalized_laplacian(adj_mx)
            .astype(np.float32)
            .todense()
        ]
    elif adj_type == "symnadj":
        adj = [
            calculate_symmetric_message_passing_adj(adj_mx).astype(np.float32).todense()
        ]
    elif adj_type == "transition":
        adj = [calculate_transition_matrix(adj_mx).T]
    elif adj_type == "doubletransition":
        adj = [
            calculate_transition_matrix(adj_mx).T,
            calculate_transition_matrix(adj_mx.T).T,
        ]
    elif adj_type == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    elif adj_type == "original":
        adj = [adj_mx]
    else:
        error = 0
        assert error, "adj type not defined"
    return adj, adj_mx
