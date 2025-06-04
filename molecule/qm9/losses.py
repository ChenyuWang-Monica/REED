import torch


def sum_except_batch(x):
    return x.view(x.size(0), -1).sum(dim=-1)


def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask)).abs().sum().item() < 1e-8


def compute_loss_and_nll(args, generative_model, nodes_dist, x, h, node_mask, edge_mask, context, rep=None, no_logpn=False):
    bs, n_nodes, n_dims = x.size()


    if args.probabilistic_model in ['diffusion', 'fm']:
        edge_mask = edge_mask.view(bs, n_nodes * n_nodes)

        assert_correctly_masked(x, node_mask)

        # Here x is a position tensor, and h is a dictionary with keys
        # 'categorical' and 'integer'.
        if generative_model.training:
            nll, denoised_xh = generative_model(x, h, node_mask, edge_mask, context, rep=rep)
        else:
            nll = generative_model(x, h, node_mask, edge_mask, context, rep=rep)
            
        N = node_mask.squeeze(2).sum(1).long()
        
        if not no_logpn:
            log_pN = nodes_dist.log_prob(N)

            assert nll.size() == log_pN.size()
            nll = nll - log_pN

        # Average over batch.
        nll = nll.mean(0)

        reg_term = torch.tensor([0.]).to(nll.device)
        mean_abs_z = 0.
    else:
        raise ValueError(args.probabilistic_model)

    if generative_model.training:
        return nll, reg_term, mean_abs_z, denoised_xh
    else:
        return nll, reg_term, mean_abs_z
        
