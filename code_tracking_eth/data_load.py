import pandas as pd
import pickle
import os
from utils import *
from graph import *


def data_load(path=data_path, graph_id=3, emb_ratio=0.7):
    pklfile_emb = path + f"LPsubG{graph_id}_df_emb_{emb_ratio}.pickle"
    pklfile_train_test_edges = path + f"LPsubG{graph_id}_train_test_edges_{emb_ratio}.pickle"

    print("Data Loading...")
    df_emb = pd.read_pickle(pklfile_emb)
    if os.path.exists(pklfile_train_test_edges):
        with open(pklfile_train_test_edges, "rb") as f:
            train_test_edges = pickle.load(f)
            train_edges_pos = train_test_edges['train_edges_pos']
            train_edges_neg = train_test_edges['train_edges_false']
            test_edges_pos = train_test_edges['test_edges_pos']
            test_edges_neg = train_test_edges['test_edges_false']

    print(f"Embedding data shape: {df_emb.shape}")
    print("Num of positive Train edges:", len(train_edges_pos))
    print("Num of positive Test edges:", len(test_edges_pos))
    print("Num of negative Train edges:", len(train_edges_neg))
    print("Num of negative Test edges:", len(test_edges_neg), "\n")

    return df_emb, train_test_edges


def static_graph_load(path=graph_path, graph_id=3, emb_ratio=0.7):
    print("Static Graph Loading...")
    with open(path + f'LPsubG{graph_id}_static_graph_with_node_features_{emb_ratio}.pkl', 'rb') as f:
        data = pickle.load(f)

    print("Num of nodes:", data.num_nodes)
    print("Dimension of node features:", data.x.shape[1])
    print("Num of edges:", data.edge_index.shape[1])
    print("Num of train edges for LP:", data.train_edge_pos_index.shape[1])
    print("Num of test edges for LP:", data.test_edge_pos_index.shape[1], "\n")

    return data


def temporal_graph_load(path=graph_path, temporal_steps=10, graph_id=3, emb_ratio=0.7, overlap_ratio=0):
    print("Temporal Graph Loading...")

    if overlap_ratio > 0:
        with open(path + f'LPsubG{graph_id}_temp_graph_{temporal_steps}_overlap_{overlap_ratio}_with_node_features_{emb_ratio}.pkl', 'rb') as f:
            data = pickle.load(f)
    else:
        with open(path + f'LPsubG{graph_id}_temp_graph_{temporal_steps}_with_node_features_{emb_ratio}.pkl', 'rb') as f:
            data = pickle.load(f)

    print("Num of temporal steps:", data.temporal_steps)
    print("Num of nodes:", [feature.shape[0] for feature in data.x])
    print("Dimension of node features:", [feature.shape[1] for feature in data.x])
    print("Num of edges:", [edge_index.shape[1] for edge_index in data.edge_indexs])
    print("Num of train edges for LP:", data.train_edge_pos_index.shape[1])
    print("Num of test edges for LP:", data.test_edge_pos_index.shape[1], "\n")

    return data


def edge_feature_load(path=data_path, temporal_steps=10, graph_id=3, emb_ratio=0.7, overlap_ratio=0):
    print("Edge Feature Loading...")

    if overlap_ratio > 0:
        train_edges_features = np.load(path + f'LPsubG{graph_id}_temp_{temporal_steps}_overlap_{overlap_ratio}_train_edges_features_{emb_ratio}.npy')
        train_edges_labels = np.load(path + f'LPsubG{graph_id}_temp_{temporal_steps}_overlap_{overlap_ratio}_train_edges_labels_{emb_ratio}.npy')
        test_edges_features = np.load(path + f'LPsubG{graph_id}_temp_{temporal_steps}_overlap_{overlap_ratio}_test_edges_features_{emb_ratio}.npy')
        test_edges_labels = np.load(path + f'LPsubG{graph_id}_temp_{temporal_steps}_overlap_{overlap_ratio}_test_edges_labels_{emb_ratio}.npy')
    else:
        train_edges_features = np.load(path + f'LPsubG{graph_id}_temp_{temporal_steps}_train_edges_features_{emb_ratio}.npy')
        train_edges_labels = np.load(path + f'LPsubG{graph_id}_temp_{temporal_steps}_train_edges_labels_{emb_ratio}.npy')
        test_edges_features = np.load(path + f'LPsubG{graph_id}_temp_{temporal_steps}_test_edges_features_{emb_ratio}.npy')
        test_edges_labels = np.load(path + f'LPsubG{graph_id}_temp_{temporal_steps}_test_edges_labels_{emb_ratio}.npy')

    print("Shape of Train Edge Features:", train_edges_features.shape)
    print("Shape of Train Edge Labels:", train_edges_labels.shape)
    print("Shape of Test Edge Features:", test_edges_features.shape)
    print("Shape of Test Edge Labels:", test_edges_labels.shape, "\n")

    return train_edges_features, train_edges_labels, test_edges_features, test_edges_labels


if __name__ == '__main__':
    edge_feature_load()
