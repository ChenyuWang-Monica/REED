
from __future__ import print_function
import numpy as np
import torch
import torch.utils
import torch.utils.checkpoint
import torch.nn as nn
import torch.nn.functional as F
import random

MASKED_TOKEN = 'Z'
ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'
ALPHABET_WITH_MASK = ALPHABET + MASKED_TOKEN
MASK_TOKEN_INDEX = ALPHABET_WITH_MASK.index(MASKED_TOKEN)


def featurize(batch, device, repr_dims, load_repr=True):
    B = len(batch)
    lengths = np.array([len(b['seq']) for b in batch], dtype=np.int32)  # sum of chain seq lengths
    L_max = max([len(b['seq']) for b in batch])
    X = np.zeros([B, L_max, 4, 3])
    residue_idx = -100 * np.ones([B, L_max], dtype=np.int32)  # residue idx with jumps across chains
    chain_M = np.zeros([B, L_max], dtype=np.int32)  # 1.0 for the bits that need to be predicted, 0.0 for the bits that are given
    mask_self = np.ones([B, L_max, L_max], dtype=np.int32)  # for interface loss calculation - 0.0 for self interaction, 1.0 for other
    chain_encoding_all = np.zeros([B, L_max], dtype=np.int32)  # integer encoding for chains 0, 0, 0,...0, 1, 1,..., 1, 2, 2, 2...
    
    repr_single = np.zeros([B, L_max, repr_dims['single']], dtype=np.float32)  # single residue representation
    repr_pair = np.zeros([B, L_max, L_max, repr_dims['pair']], dtype=np.float32)  # pairwise residue representation
    repr_structure = np.zeros([B, L_max, repr_dims['structure']], dtype=np.float32)  # structure representation

    S = np.zeros([B, L_max], dtype=np.int32)  # sequence AAs integers
    init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                     'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                     'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    extra_alphabet = [str(item) for item in list(np.arange(300))]
    chain_letters = init_alphabet + extra_alphabet
    for i, b in enumerate(batch):
        masked_chains = b['masked_list']
        visible_chains = b['visible_list']
        all_chains = masked_chains + visible_chains
        visible_temp_dict = {}
        masked_temp_dict = {}
        for step, letter in enumerate(all_chains):
            chain_seq = b[f'seq_chain_{letter}']
            if letter in visible_chains:
                visible_temp_dict[letter] = chain_seq
            elif letter in masked_chains:
                masked_temp_dict[letter] = chain_seq
        for km, vm in masked_temp_dict.items():
            for kv, vv in visible_temp_dict.items():
                if vm == vv:
                    if kv not in masked_chains:
                        masked_chains.append(kv)
                    if kv in visible_chains:
                        visible_chains.remove(kv)
        all_chains = masked_chains + visible_chains
        random.shuffle(all_chains)  # randomly shuffle chain order
        num_chains = b['num_of_chains']
        mask_dict = {}
        x_chain_list = []
        chain_mask_list = []
        chain_seq_list = []
        chain_encoding_list = []
        repr_single_list = []
        repr_pair_list = []
        repr_structure_list = []
        c = 1
        l0 = 0
        l1 = 0
        for step, letter in enumerate(all_chains):
            if letter in visible_chains:
                chain_seq = b[f'seq_chain_{letter}']
                chain_length = len(chain_seq)
                chain_coords = b[f'coords_chain_{letter}']  # this is a dictionary
                chain_mask = np.zeros(chain_length)  # 0.0 for visible chains
                x_chain = np.stack([chain_coords[c] for c in
                                    [f'N_chain_{letter}', f'CA_chain_{letter}', f'C_chain_{letter}',
                                     f'O_chain_{letter}']], 1)  # [chain_length,4,3]
                x_chain_list.append(x_chain)
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(c * np.ones(np.array(chain_mask).shape[0]))
                # chain_encoding_list.append(c * torch.ones(chain_mask.shape[0], dtype=torch.long, device=device))
                if load_repr:
                    repr_single_list.append(b[f'repr_single_{letter}'])#.to(dtype=torch.float32, device=device))
                    repr_pair_list.append(b[f'repr_pair_{letter}'])#.to(dtype=torch.float32, device=device))
                    repr_structure_list.append(b[f'repr_structure_{letter}'])#.to(dtype=torch.float32, device=device))

                l1 += chain_length
                mask_self[i, l0:l1, l0:l1] = np.zeros([chain_length, chain_length])
                residue_idx[i, l0:l1] = 100 * (c - 1) + np.arange(l0, l1)
                l0 += chain_length
                c += 1
            elif letter in masked_chains:
                chain_seq = b[f'seq_chain_{letter}']
                chain_length = len(chain_seq)
                chain_coords = b[f'coords_chain_{letter}']  # this is a dictionary
                chain_mask = np.ones(chain_length)  # 0.0 for visible chains
                # chain_mask = torch.ones(chain_length, dtype=torch.int64, device=device)
                x_chain = np.stack([chain_coords[c] for c in
                                    [f'N_chain_{letter}', f'CA_chain_{letter}', f'C_chain_{letter}',
                                     f'O_chain_{letter}']], 1)  # [chain_lenght,4,3]
                x_chain_list.append(x_chain)
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(c * np.ones(np.array(chain_mask).shape[0]))
                if load_repr:
                    repr_single_list.append(b[f'repr_single_{letter}'])
                    repr_pair_list.append(b[f'repr_pair_{letter}'])
                    repr_structure_list.append(b[f'repr_structure_{letter}'])

                l1 += chain_length
                mask_self[i, l0:l1, l0:l1] = np.zeros([chain_length, chain_length])
                # mask_self[i, l0:l1, l0:l1] = torch.zeros([chain_length, chain_length], dtype=torch.int64, device=device)
                residue_idx[i, l0:l1] = 100 * (c - 1) + np.arange(l0, l1)
                # residue_idx[i, l0:l1] = 100 * (c - 1) + torch.arange(l0, l1, device=device).long()
                l0 += chain_length
                c += 1
        x = np.concatenate(x_chain_list, 0)  # [L, 4, 3]
        all_sequence = "".join(chain_seq_list)
        m = np.concatenate(chain_mask_list, 0)  # [L,], 1.0 for places that need to be predicted
        chain_encoding = np.concatenate(chain_encoding_list, 0)

        l = len(all_sequence)
        X[i, :l, :, :] = x

        chain_M[i, :l] = m

        chain_encoding_all[i, :l] = chain_encoding

        # pad repr_single, repr_pair, repr_structure, they have shape [L, hidden_dim], [L, L, hidden_dim], [L, hidden_dim]
        if load_repr:
            repr_s = np.concatenate(repr_single_list, 0)
            repr_p = np.concatenate(repr_pair_list, 0)
            repr_st = np.concatenate(repr_structure_list, 0)
            repr_single[i, :l, :] = repr_s
            repr_pair[i, :l, :l, :] = repr_p
            repr_structure[i, :l, :] = repr_st

        indices = np.asarray([ALPHABET.index(a) for a in all_sequence], dtype=np.int32)
        S[i, :l] = indices

    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X, (2, 3))).astype(np.float32)
    X[isnan] = 0.

    # Conversion
    residue_idx = torch.from_numpy(residue_idx).to(dtype=torch.long, device=device)
    S = torch.from_numpy(S).to(dtype=torch.long, device=device)
    X = torch.from_numpy(X).to(dtype=torch.float32, device=device)
    mask = torch.from_numpy(mask).to(dtype=torch.float32, device=device)
    mask_self = torch.from_numpy(mask_self).to(dtype=torch.float32, device=device)
    chain_M = torch.from_numpy(chain_M).to(dtype=torch.float32, device=device)
    chain_encoding_all = torch.from_numpy(chain_encoding_all).to(dtype=torch.long, device=device)
    if load_repr:
        repr_single = torch.from_numpy(repr_single).to(dtype=torch.float32, device=device)
        repr_pair = torch.from_numpy(repr_pair).to(dtype=torch.float32, device=device)
        repr_structure = torch.from_numpy(repr_structure).to(dtype=torch.float32, device=device)
    else:
        repr_single = None
        repr_pair = None
        repr_structure = None

    return X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all, repr_single, repr_pair, repr_structure


def loss_nll(S, log_probs, mask):
    """ Negative log probabilities """
    criterion = torch.nn.NLLLoss(reduction='none')
    loss = criterion(
        log_probs.contiguous().view(-1, log_probs.size(-1)), S.contiguous().view(-1)
    ).view(S.size())
    S_argmaxed = torch.argmax(log_probs, -1)  # [B, L]
    true_false = (S == S_argmaxed).float()
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av, true_false


def loss_smoothed(S, log_probs, mask, weight=0.1, t=None, t_schedule=False):
    """ Negative log probabilities """
    S_onehot = torch.nn.functional.one_hot(S, 22).float()

    # Label smoothing
    S_onehot = S_onehot + weight / float(S_onehot.size(-1))
    S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)

    loss = -(S_onehot * log_probs).sum(-1)
    if t_schedule:
        assert t is not None
        # t has the shape [B]
        loss = loss / (1-t).unsqueeze(-1) / 4.605  # log(0.01) = -4.605
    loss_av = torch.sum(loss * mask) / 2000.0  # fixed
    return loss, loss_av


def loss_repr(repr_single, repr_pair, repr_structure, zs_single, zs_pair, zs_structure, E_idx,
              mask, mask_residue, repr_norm=False, repr_noise=0.0):
    repr_single = repr_single.to(torch.float32)
    repr_pair = repr_pair.to(torch.float32)
    repr_structure = repr_structure.to(torch.float32)/100.0
    # cosine similarity loss
    B, L, L, z = repr_pair.size()
    B1, L1, num_edges = E_idx.size()
    assert B == B1 and L == L1
    # take the indices of edges and gather the representations of repr_pair according to E_idx
    repr_pair = gather_edges(repr_pair, E_idx)
    mask_pair = (mask.unsqueeze(-1) * mask.unsqueeze(1)).unsqueeze(-1)  # [B, L, L, 1]
    mask_pair = gather_edges(mask_pair, E_idx).squeeze()  # [B, L, K]
    # change [B,L] mask_residue to [B,L,L] mask_residue which is the repeat of mask_residue
    mask_pair_residue = mask_residue.unsqueeze(-1).repeat(1, 1, L).unsqueeze(-1)
    mask_pair_residue = gather_edges(mask_pair_residue, E_idx).squeeze()
    mask_pair_residue = mask_pair_residue * mask_pair

    # mean-std normalization of z
    # z = (z - z.mean(dim=0, keepdim=True)) / (z.std(dim=0, keepdim=True) + 1e-6)
    if repr_norm:
        repr_single_mean = (repr_single * mask.unsqueeze(-1)).sum((0, 1), keepdim=True) / mask.sum()
        repr_single_std = (((repr_single * mask.unsqueeze(-1) - repr_single_mean) ** 2).sum((0, 1),
                                                                                            keepdim=True) / mask.sum()) ** 0.5
        repr_single = (repr_single - repr_single_mean) / (repr_single_std + 1e-6)
        repr_pair_mean = (repr_pair * mask_pair.unsqueeze(-1)).sum((0, 1, 2), keepdim=True) / mask_pair.sum()
        repr_pair_std = (((repr_pair * mask_pair.unsqueeze(-1) - repr_pair_mean) ** 2).sum((0, 1, 2),
                                                                                           keepdim=True) / mask_pair.sum()) ** 0.5
        repr_pair = (repr_pair - repr_pair_mean) / (repr_pair_std + 1e-6)
        repr_structure_mean = (repr_structure * mask.unsqueeze(-1)).sum((0, 1), keepdim=True) / mask.sum()
        repr_structure_std = (((repr_structure * mask.unsqueeze(-1) - repr_structure_mean) ** 2).sum((0, 1),
                                                                                                     keepdim=True) / mask.sum()) ** 0.5
        repr_structure = (repr_structure - repr_structure_mean) / (repr_structure_std + 1e-6)

    repr_single = repr_single + torch.randn_like(repr_single, dtype=repr_single.dtype, device=repr_single.device) * repr_noise
    repr_pair = repr_pair + torch.randn_like(repr_pair, dtype=repr_pair.dtype, device=repr_pair.device) * repr_noise
    repr_structure = repr_structure + torch.randn_like(repr_structure, dtype=repr_structure.dtype, device=repr_structure.device) * repr_noise

    repr_single = F.normalize(repr_single, p=2, dim=-1)
    repr_pair = F.normalize(repr_pair.view(B, -1, z), p=2, dim=-1)
    repr_structure = F.normalize(repr_structure, p=2, dim=-1)
    zs_single = F.normalize(zs_single, p=2, dim=-1)
    zs_pair = F.normalize(zs_pair.view(B, -1, z), p=2, dim=-1)
    zs_structure = F.normalize(zs_structure, p=2, dim=-1)

    loss_single = -((repr_single * zs_single).sum(-1) * mask).sum() / torch.sum(mask)
    loss_pair = -(repr_pair * zs_pair).sum(-1)  # [B, L*30]
    loss_pair = (loss_pair * mask_pair.view(B, -1)).sum() / torch.sum(mask_pair)
    loss_structure = -((repr_structure * zs_structure).sum(-1) * mask).sum() / torch.sum(mask)
    loss_single_masked = -((repr_single * zs_single).sum(-1) * mask_residue).sum() / torch.sum(mask_residue)
    loss_pair_masked = -(repr_pair * zs_pair).sum(-1)  # [B, L*30]
    loss_pair_masked = (loss_pair_masked * mask_pair_residue.view(B, -1)).sum() / torch.sum(mask_pair_residue)
    loss_structure_masked = -((repr_structure * zs_structure).sum(-1) * mask_residue).sum() / torch.sum(mask_residue)


    return loss_single, loss_pair, loss_structure, loss_single_masked, loss_pair_masked, loss_structure_masked


# The following gather functions
def gather_edges(edges, neighbor_idx):
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features


def gather_nodes(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features


def gather_nodes_t(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor index [B,K] => Neighbor features[B,K,C]
    idx_flat = neighbor_idx.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, idx_flat)
    return neighbor_features


def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn


def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
        nn.Linear(hidden_size, projector_dim),
        # nn.SiLU(),
        # nn.Linear(projector_dim, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, z_dim),
    )


class EncLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(EncLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message))
        return h_V, h_E


class DecLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(DecLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)
        self.W_edge = nn.Sequential(nn.LayerNorm(3 * num_hidden), nn.Linear(3 * num_hidden, 3 * num_hidden),
                                    nn.GELU(), nn.Linear(3 * num_hidden, 2 * num_hidden))
        self.dropout_edge = nn.Dropout(dropout)
        self.norm_edge = nn.LayerNorm(2 * num_hidden)

    def forward(self, h_V, h_E, h_ES, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """

        # Concatenate h_V_i to h_E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_E], -1)

        # h_EV: [batch_size, seq_length, num_edge, 4 * hidden_dim]

        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dE = self.W_edge(torch.cat([h_message, h_ES], dim=-1))
        h_ES_new = self.norm_edge(h_ES + self.dropout_edge(dE))
        dh = torch.sum(h_message, -2) / self.scale

        h_V = self.norm1(h_V + self.dropout1(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V, h_ES_new


class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)
        self.act = torch.nn.GELU()

    def forward(self, h_V):
        h = self.act(self.W_in(h_V))
        h = self.W_out(h)
        return h


class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings, max_relative_feature=32):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2 * max_relative_feature + 1 + 1, num_embeddings)

    def forward(self, offset, mask):
        d = torch.clip(offset + self.max_relative_feature, 0, 2 * self.max_relative_feature) * mask + (1 - mask) * (
                    2 * self.max_relative_feature + 1)
        d_onehot = torch.nn.functional.one_hot(d, 2 * self.max_relative_feature + 1 + 1)
        E = self.linear(d_onehot.float())
        return E


class ProteinFeatures(nn.Module):
    def __init__(self, node_features, edge_features, num_positional_embeddings=16,
                 num_rbf=16, top_k=30, augment_eps=0., num_chain_embeddings=16):
        """ Extract protein features """
        super(ProteinFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        self.embeddings = PositionalEncodings(num_positional_embeddings)
        node_in, edge_in = 6, num_positional_embeddings + num_rbf * 25
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = nn.LayerNorm(edge_features)

    def _dist(self, X, mask, eps=1E-6):
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = mask_2D * torch.sqrt(torch.sum(dX ** 2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        sampled_top_k = self.top_k
        D_neighbors, E_idx = torch.topk(D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False)
        return D_neighbors, E_idx

    def _rbf(self, D):
        device = D.device
        D_min, D_max, D_count = 2., 22., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1, 1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(torch.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, -1) + 1e-6)  # [B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:, :, :, None], E_idx)[:, :, :, 0]  # [B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def forward(self, X, mask, residue_idx, chain_labels):
        if self.training and self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)

        b = X[:, :, 1, :] - X[:, :, 0, :]
        c = X[:, :, 2, :] - X[:, :, 1, :]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + X[:, :, 1, :]
        Ca = X[:, :, 1, :]
        N = X[:, :, 0, :]
        C = X[:, :, 2, :]
        O = X[:, :, 3, :]

        D_neighbors, E_idx = self._dist(Ca, mask)

        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors))  # Ca-Ca
        RBF_all.append(self._get_rbf(N, N, E_idx))  # N-N
        RBF_all.append(self._get_rbf(C, C, E_idx))  # C-C
        RBF_all.append(self._get_rbf(O, O, E_idx))  # O-O
        RBF_all.append(self._get_rbf(Cb, Cb, E_idx))  # Cb-Cb
        RBF_all.append(self._get_rbf(Ca, N, E_idx))  # Ca-N
        RBF_all.append(self._get_rbf(Ca, C, E_idx))  # Ca-C
        RBF_all.append(self._get_rbf(Ca, O, E_idx))  # Ca-O
        RBF_all.append(self._get_rbf(Ca, Cb, E_idx))  # Ca-Cb
        RBF_all.append(self._get_rbf(N, C, E_idx))  # N-C
        RBF_all.append(self._get_rbf(N, O, E_idx))  # N-O
        RBF_all.append(self._get_rbf(N, Cb, E_idx))  # N-Cb
        RBF_all.append(self._get_rbf(Cb, C, E_idx))  # Cb-C
        RBF_all.append(self._get_rbf(Cb, O, E_idx))  # Cb-O
        RBF_all.append(self._get_rbf(O, C, E_idx))  # O-C
        RBF_all.append(self._get_rbf(N, Ca, E_idx))  # N-Ca
        RBF_all.append(self._get_rbf(C, Ca, E_idx))  # C-Ca
        RBF_all.append(self._get_rbf(O, Ca, E_idx))  # O-Ca
        RBF_all.append(self._get_rbf(Cb, Ca, E_idx))  # Cb-Ca
        RBF_all.append(self._get_rbf(C, N, E_idx))  # C-N
        RBF_all.append(self._get_rbf(O, N, E_idx))  # O-N
        RBF_all.append(self._get_rbf(Cb, N, E_idx))  # Cb-N
        RBF_all.append(self._get_rbf(C, Cb, E_idx))  # C-Cb
        RBF_all.append(self._get_rbf(O, Cb, E_idx))  # O-Cb
        RBF_all.append(self._get_rbf(C, O, E_idx))  # C-O
        RBF_all = torch.cat(tuple(RBF_all), dim=-1)

        offset = residue_idx[:, :, None] - residue_idx[:, None, :]
        offset = gather_edges(offset[:, :, :, None], E_idx)[:, :, :, 0]  # [B, L, K]

        d_chains = ((chain_labels[:, :, None] - chain_labels[:, None,
                                                :]) == 0).long()  # find self vs non-self interaction
        E_chains = gather_edges(d_chains[:, :, :, None], E_idx)[:, :, :, 0]
        E_positional = self.embeddings(offset.long(), E_chains)
        E = torch.cat((E_positional, RBF_all), -1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)
        return E, E_idx


class ProteinMPNNFMIF(nn.Module):
    def __init__(self, node_features=128, edge_features=128,
                 hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
                 vocab=22, k_neighbors=32, augment_eps=0.1, dropout=0.1, cfg=False,
                 update_edge=False, align_depth=1, learnable_node=False, projector_dim=1024,
                 single_dim=384, pair_dim=128, structure_dim=768, mdlm_parameterization=False):
        super(ProteinMPNNFMIF, self).__init__()

        # Hyperparameters
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.vocab = vocab
        self.update_edge = update_edge
        self.align_depth = align_depth
        self.mdlm_parameterization = mdlm_parameterization

        self.features = ProteinFeatures(node_features, edge_features, top_k=k_neighbors, augment_eps=augment_eps)
        self.node_embeds = nn.Parameter(torch.zeros(hidden_dim))
        if not learnable_node:
            self.node_embeds.requires_grad_(False)

        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_s = nn.Embedding(vocab, hidden_dim)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncLayer(hidden_dim, hidden_dim * 2, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecLayer(hidden_dim, hidden_dim * 3, dropout=dropout)
            for _ in range(num_decoder_layers)
        ])
        self.W_out = nn.Linear(hidden_dim, vocab, bias=True)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        if cfg:
            self.cls_embedder = nn.Embedding(num_embeddings=2 + 1, embedding_dim=hidden_dim)
            self.cls_layers_enc = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_encoder_layers)])
            self.cls_layers_dec = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_decoder_layers)])

        self.projectors_single = build_mlp(hidden_dim, single_dim * 2, single_dim)
        self.projectors_pair = build_mlp(hidden_dim * 2, pair_dim * 2, pair_dim)
        self.projectors_structure = build_mlp(hidden_dim, structure_dim * 2, structure_dim)

    def finetune_init(self):
        self.W_s_ft = nn.Linear(self.vocab, self.hidden_dim, bias=False).cuda()
        self.W_s_ft.weight = nn.Parameter(self.W_s.weight.data.T)

    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all, cls=None, return_rep=False,
                detach_repr=False):
        """ Graph-conditioned sequence model """
        # X: [batch_size, seq_length, 4, 3]; S: [batch_size, seq_length], same for mask/chain_M/residue_idx
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        # h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        # Cai: we use learnable node initialization features instead of zero initialization
        h_V = self.node_embeds.unsqueeze(0).unsqueeze(0).repeat(E.shape[0], E.shape[1], 1)
        h_E = self.W_e(E)
        # h_V: [batch_size, seq_length, hidden_dim]
        # E, h_E: [batch_size, seq_length, num_edge, (edge_features=)hidden_dim]
        # E_idx: [batch_size, seq_length, num_edge]

        if cls is not None:
            cls_emb = self.cls_embedder(cls)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for i, layer in enumerate(self.encoder_layers):
            if cls is not None:
                h_V = h_V + self.cls_layers_enc[i](cls_emb)[:, None, :]
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        zs_structure = h_V

        if S.ndim > 2 and S.shape[-1] == 22:
            h_S = self.W_s_ft(S)
        else:
            h_S = self.W_s(S)

        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)
        chain_M = chain_M * mask  # update chain_M to include missing regions
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        # h_S: [batch_size, seq_length, hidden_dim]
        # h_ES: [batch_size, seq_length, num_edge, 2 * hidden_dim]

        # zs_single = []
        # zs_pair = []
        for i, layer in enumerate(self.decoder_layers):
            if cls is not None:
                h_V = h_V + self.cls_layers_dec[i](cls_emb)[:, None, :]
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_1D * h_ESV
            h_V, h_ES_new = layer(h_V, h_ESV, h_ES, mask)
            if self.update_edge:
                h_ES = h_ES_new
                # Cai: updating edge improves the performance
            if i == self.align_depth:
                zs_single = h_V
                zs_pair = h_ES_new  # whether update edge or not, the align object of pair representation is h_ES_new
            # h_V: [batch_size, seq_length, hidden_dim]
            # h_ESV: [batch_size, seq_length, num_edge, 3 * hidden_dim]

        logits = self.W_out(h_V)
        if self.mdlm_parameterization:
            logits[:, :, MASK_TOKEN_INDEX] = -1e6
            log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
            unmasked_indices = (S != MASK_TOKEN_INDEX)
            log_probs[unmasked_indices] = -1e6
            log_probs[unmasked_indices, S[unmasked_indices]] = 0
        else:
            log_probs = F.log_softmax(logits, dim=-1)

        if detach_repr:
            zs_single = zs_single.detach()
            zs_pair = zs_pair.detach()
            zs_structure = zs_structure.detach()
        zs_single = self.projectors_single(zs_single.view(-1, zs_single.size(-1))).view(zs_single.size(0),
                                                                                        zs_single.size(1), -1)
        zs_pair = self.projectors_pair(zs_pair.view(-1, zs_pair.size(-1))).view(zs_pair.size(0), zs_pair.size(1),
                                                                                zs_pair.size(2), -1)
        zs_structure = self.projectors_structure(zs_structure.view(-1, zs_structure.size(-1))).view(
            zs_structure.size(0), zs_structure.size(1), -1)

        if return_rep:
            return log_probs, zs_single, zs_pair, zs_structure, E_idx
        return log_probs


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer, step):
        self.optimizer = optimizer
        self._step = step
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    @property
    def param_groups(self):
        """Return param_groups."""
        return self.optimizer.param_groups

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()


def get_std_opt(parameters, d_model, step):
    return NoamOpt(
        d_model, 2, 4000, torch.optim.Adam(parameters, lr=0, betas=(0.9, 0.98), eps=1e-9), step
    )
