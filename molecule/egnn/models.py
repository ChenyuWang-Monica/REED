import torch
import torch.nn as nn
from egnn.egnn_new import EGNN, GNN
from equivariant_diffusion.utils import remove_mean, remove_mean_with_mask
import numpy as np
from equivariant_diffusion.utils import assert_correctly_masked
from equivariant_diffusion import utils as diffusion_utils


class EGNN_dynamics_QM9(nn.Module):
    def __init__(self, in_node_nf, context_node_nf,
                 n_dims, hidden_nf=64, device='cpu',
                 act_fn=torch.nn.SiLU(), n_layers=4, attention=False,
                 condition_time=True, tanh=False, mode='egnn_dynamics', norm_constant=0,
                 inv_sublayers=2, sin_embedding=False, normalization_factor=100, aggregation_method='sum', rep_dropout_prob=None, cfg=None, rep_nf=None, attn_dropout=None, attn_block_num=1):
        super().__init__()
        
        
        self.context_node_nf = context_node_nf
        
        assert rep_dropout_prob is not None
        self.rep_dropout_prob = rep_dropout_prob
        if rep_dropout_prob > 0:
            self.fake_latent = nn.Parameter(torch.zeros(1, rep_nf))
            torch.nn.init.normal_(self.fake_latent, std=.02)
        
        self.mode = mode
        if mode == 'egnn_dynamics':
            self.egnn = EGNN(
                in_node_nf=in_node_nf + context_node_nf, in_edge_nf=1,
                hidden_nf=hidden_nf, device=device, act_fn=act_fn,
                n_layers=n_layers, attention=attention, tanh=tanh, norm_constant=norm_constant,
                inv_sublayers=inv_sublayers, sin_embedding=sin_embedding,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method, rep_nf=rep_nf, attn_dropout=attn_dropout, attn_block_num=attn_block_num)
            self.in_node_nf = in_node_nf
        elif mode == 'gnn_dynamics':
            assert 0, "Not used."
            self.gnn = GNN(
                in_node_nf=in_node_nf + context_node_nf + 3, in_edge_nf=0,
                hidden_nf=hidden_nf, out_node_nf=3 + in_node_nf, device=device,
                act_fn=act_fn, n_layers=n_layers, attention=attention,
                normalization_factor=normalization_factor, aggregation_method=aggregation_method)
            

        self.device = device
        self.n_dims = n_dims
        self._edges_dict = {}
        self.condition_time = condition_time
        self.cfg = cfg

    def forward(self, t, xh, node_mask, edge_mask, context=None):
        raise NotImplementedError

    def wrap_forward(self, node_mask, edge_mask, context):
        def fwd(time, state):
            return self._forward(time, state, node_mask, edge_mask, context)
        return fwd

    def unwrap_forward(self):
        return self._forward
    
    
    def forward_with_cfg(self, t, xh, node_mask, edge_mask, context, rep=None):
        assert not self.training and not self.egnn.training
        

        batch_size, max_node_num = t.shape[0], xh.shape[1]
        t = torch.cat([t,t])
        xh = torch.cat([xh, xh])
        half_node_mask = node_mask
        node_mask = torch.cat([node_mask, node_mask])
        edge_mask = torch.cat([edge_mask, edge_mask])
        assert self.mode == 'egnn_dynamics'
        rep = torch.cat([rep, self.fake_latent.repeat(batch_size, 1)])
        updated_xh = self._forward( t, xh, node_mask, edge_mask, context, rep=rep)
        cond, uncond = torch.split(updated_xh, batch_size) 
        assert cond.shape == uncond.shape
        updated_xh_cfg = cond + self.cfg * (cond - uncond) 
        
        assert_correctly_masked(updated_xh_cfg, half_node_mask)
        diffusion_utils.assert_mean_zero_with_mask(updated_xh_cfg[:, :, :self.n_dims], half_node_mask)
        return updated_xh_cfg
        

    def _forward(self, t, xh, node_mask, edge_mask, context, rep=None):
        bs, n_nodes, dims = xh.shape
        h_dims = dims - self.n_dims
        edges = self.get_adj_matrix(n_nodes, bs, self.device)
        edges = [x.to(self.device) for x in edges]
        node_mask = node_mask.view(bs*n_nodes, 1)
        edge_mask = edge_mask.view(bs*n_nodes*n_nodes, 1)
        xh = xh.view(bs*n_nodes, -1).clone() * node_mask
        x = xh[:, 0:self.n_dims].clone()
        if h_dims == 0:
            h = torch.ones(bs*n_nodes, 1).to(self.device)
        else:
            h = xh[:, self.n_dims:].clone()

        if self.condition_time:
            if np.prod(t.size()) == 1:
                # t is the same for all elements in batch.
                h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
            else:
                # t is different over the batch dimension.
                h_time = t.view(bs, 1).repeat(1, n_nodes)
                h_time = h_time.view(bs * n_nodes, 1)
            h = torch.cat([h, h_time], dim=1)

        assert rep is not None

        if context is not None:
            # We're conditioning, awesome!
            context = context.view(bs*n_nodes, -1)
            context = context * node_mask
            h = torch.cat([h, context], dim=1)
            
            
        if self.rep_dropout_prob > 0.0 and self.training:
            drop_rep_mask = (torch.rand(bs) < self.rep_dropout_prob).unsqueeze(-1).to(self.fake_latent.device).to(torch.float32)
            rep = drop_rep_mask * self.fake_latent.repeat(rep.shape[0], 1) + (1 - drop_rep_mask) * rep
        

        if self.mode == 'egnn_dynamics':
            h_final, x_final = self.egnn(h, x, edges, node_mask=node_mask, edge_mask=edge_mask, rep=rep)
            vel = (x_final - x) * node_mask  # This masking operation is redundant but just in case
        elif self.mode == 'gnn_dynamics':
            xh = torch.cat([x, h], dim=1)
            output = self.gnn(xh, edges, node_mask=node_mask)
            vel = output[:, 0:3] * node_mask
            h_final = output[:, 3:]

        else:
            raise Exception("Wrong mode %s" % self.mode)

        if context is not None:
            # Slice off context size:
            h_final = h_final[:, :-self.context_node_nf]

        if self.condition_time:
            # Slice off last dimension which represented time.
            h_final = h_final[:, :-1]

        vel = vel.view(bs, n_nodes, -1)

        if torch.any(torch.isnan(vel)):
            print('Warning: detected nan, resetting EGNN output to zero.')
            vel = torch.zeros_like(vel)

        if node_mask is None:
            vel = remove_mean(vel)
        else:
            vel = remove_mean_with_mask(vel, node_mask.view(bs, n_nodes, 1))

        if h_dims == 0:
            return vel
        else:
            h_final = h_final.view(bs, n_nodes, -1)
            return torch.cat([vel, h_final], dim=2)

    def get_adj_matrix(self, n_nodes, batch_size, device):
        if n_nodes in self._edges_dict:
            edges_dic_b = self._edges_dict[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                # get edges for a single sample
                rows, cols = [], []
                for batch_idx in range(batch_size):  # XXX: Optimize this!!
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            rows.append(i + batch_idx * n_nodes)
                            cols.append(j + batch_idx * n_nodes)
                edges = [torch.LongTensor(rows).to(device),
                         torch.LongTensor(cols).to(device)]
                edges_dic_b[batch_size] = edges
                return edges
        else:
            self._edges_dict[n_nodes] = {}
            return self.get_adj_matrix(n_nodes, batch_size, device)
