import sys, os
sys.path.append("../../")
from qm9.property_prediction.models_property import EGNN, Naive, NumNodes
import torch
from torch import nn, optim
import argparse
from qm9.property_prediction import prop_utils
import json
from qm9 import dataset, utils
import pickle
from models.encoders import initialize_encoder, get_global_representation
import numpy as np
loss_l1 = nn.L1Loss()


def train(model, epoch, loader, mean, mad, property, device, partition='train', optimizer=None, lr_scheduler=None, log_interval=20, debug_break=False, output_molecules=False):
    if partition == 'train':
        lr_scheduler.step()
        loader.sampler.set_epoch(epoch)
    res = {'loss': 0, 'counter': 0, 'loss_arr':[]}
    if output_molecules:
        molecules = {'one_hot': [], 'x': [], 'node_mask': []}
        
    pairs = []
        
    for i, data in enumerate(loader):
        if partition == 'train':
            model.train()
            optimizer.zero_grad()

        else:
            model.eval()
        if output_molecules:
            molecules['one_hot'].append(data['one_hot'].cpu())
            molecules['x'].append(data['positions'].cpu())
            molecules['node_mask'].append(data['atom_mask'].unsqueeze(-1).cpu())

        
        batch_size, n_nodes, _ = data['positions'].size()
        atom_positions = data['positions'].view(batch_size * n_nodes, -1).to(device, torch.float32)
        atom_mask = data['atom_mask'].view(batch_size * n_nodes, -1).to(device, torch.float32)
        edge_mask = data['edge_mask'].to(device, torch.float32)
        nodes = data['one_hot'].to(device, torch.float32)
        #charges = data['charges'].to(device, dtype).squeeze(2)
        #nodes = prop_utils.preprocess_input(one_hot, charges, args.charge_power, charge_scale, device)

        nodes = nodes.view(batch_size * n_nodes, -1)
        # nodes = torch.cat([one_hot, charges], dim=1)
        edges = prop_utils.get_adj_matrix(n_nodes, batch_size, device)
        label = data[property].to(device, torch.float32)

        '''
        print("Positions mean")
        print(torch.mean(torch.abs(atom_positions)))
        print("Positions max")
        print(torch.max(atom_positions))
        print("Positions min")
        print(torch.min(atom_positions))


        print("\nOne hot mean")
        print(torch.mean(torch.abs(nodes)))
        print("one_hot max")
        print(torch.max(nodes))
        print("one_hot min")
        print(torch.min(nodes))


        print("\nLabel mean")
        print(torch.mean(torch.abs(label)))
        print("label max")
        print(torch.max(label))
        print("label min")
        print(torch.min(label))
        '''

        pred = model(h0=nodes, x=atom_positions, edges=edges, edge_attr=None, node_mask=atom_mask, edge_mask=edge_mask,
                     n_nodes=n_nodes)


        # print("\nPred mean")
        # print(torch.mean(torch.abs(pred)))
        # print("Pred max")
        # print(torch.max(pred))
        # print("Pred min")
        # print(torch.min(pred))

        if partition == 'train':
            loss = loss_l1(pred, (label - mean) / mad)
            loss.backward()
            optimizer.step()
        else:
            loss = loss_l1(mad * pred + mean, label)
            
            adjusted_pred = mad * pred + mean  # Adjust the prediction
            
            # Save the (mad * pred + mean, label) pairs
            for p, l in zip(adjusted_pred.cpu().detach().numpy(), label.cpu().detach().numpy()):
                pairs.append((p, l))  # Collect each (pred, label) pair

        res['loss'] += loss.item() * batch_size
        res['counter'] += batch_size
        res['loss_arr'].append(loss.item())

        prefix = ""
        if partition != 'train':
            prefix = ">> %s \t" % partition

        if i % log_interval == 0:
            print(prefix + "Epoch %d \t Iteration %d \t loss %.4f" % (epoch, i, sum(res['loss_arr'][-10:])/len(res['loss_arr'][-10:])))
        if debug_break:
            break
    
    if partition != 'train':
        pairs_file = f"mad_pred_mean_label_pairs_epoch_{epoch}.txt"
        with open(pairs_file, 'w') as f:
            f.write("Pred, Label\n")
            for pred, label in pairs:
                f.write(f"{pred}, {label}\n")
        print(f"Saved (mad * pred + mean, label) pairs to {pairs_file}")
    
    if output_molecules:
        molecules = {key: torch.cat(molecules[key], dim=0) for key in molecules}
        return res['loss'] / res['counter'], molecules
    return res['loss'] / res['counter']


def test(model, epoch, loader, mean, mad, property, device, log_interval, debug_break=False, output_molecules=False):
    return train(model, epoch, loader, mean, mad, property, device, partition='test', log_interval=log_interval, debug_break=debug_break, output_molecules=output_molecules)


def get_model(args):
    if args.model_name == 'egnn':
        model = EGNN(in_node_nf=5, in_edge_nf=0, hidden_nf=args.nf, device=args.device, n_layers=args.n_layers,
                     coords_weight=1.0,
                     attention=args.attention, node_attr=args.node_attr)
    elif args.model_name == 'naive':
        model = Naive(device=args.device)
    elif args.model_name == 'numnodes':
        model = NumNodes(device=args.device)
    elif args.model_name == "frad":
        os.chdir("../../")
        model = initialize_encoder(encoder_type="frad", device="cuda", encoder_ckpt_path="./checkpoints/encoder_ckpts/QM9.ckpt")
        os.chdir("qm9/property_prediction")
        class Encoder_Linear(nn.Module):
            def __init__(self, encoder):
                super().__init__()
                self.linear = nn.Linear(in_features=256, out_features=1)
                self.encoder = encoder
                for param in encoder.parameters():
                    param.requires_grad = False
                self.encoder.eval()
            def forward(self, h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes):
                batch_size = int(x.shape[0] / n_nodes)
                x = x.view(batch_size, n_nodes, 3)
                h0 = h0.view(batch_size, n_nodes, 5)
                node_mask = node_mask.view(batch_size, n_nodes)
                dict = {
                    0:1,
                    1:6,
                    2:7,
                    3:8,
                    4:9
                }
                h0_max_idx = torch.argmax(h0, dim=-1)
                h0 = np.vectorize(lambda x: dict[x])(h0_max_idx.cpu().numpy())
                h0 = torch.tensor(h0, device=x.device, dtype=x.dtype).unsqueeze(-1)
                
                rep = get_global_representation(node_mask, self.encoder, x, h0, training_encoder=False, noise_sigma=0., device=None)
                return self.linear(rep).squeeze()
        return Encoder_Linear(model).to(device)
    else:
        raise Exception("Wrong model name %s" % args.model_name)


    return model

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='QM9 Example')
    parser.add_argument('--exp_name', type=str, default='debug', metavar='N',
                        help='experiment_name')
    parser.add_argument('--batch_size', type=int, default=96, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 10)')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=20, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--test_interval', type=int, default=1, metavar='N',
                        help='how many epochs to wait before logging test')
    parser.add_argument('--outf', type=str, default='outputs', metavar='N',
                        help='folder to output vae')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='N',
                        help='learning rate')
    parser.add_argument('--nf', type=int, default=128, metavar='N',
                        help='learning rate')
    parser.add_argument('--attention', type=int, default=1, metavar='N',
                        help='attention in the ae model')
    parser.add_argument('--n_layers', type=int, default=7, metavar='N',
                        help='number of layers for the autoencoder')
    parser.add_argument('--property', type=str, default='alpha', metavar='N',
                        help='label to predict: alpha | gap | homo | lumo | mu | Cv | G | H | r2 | U | U0 | zpve')
    parser.add_argument('--num_workers', type=int, default=0, metavar='N',
                        help='number of workers for the dataloader')
    parser.add_argument('--filter_n_atoms', type=int, default=None,
                        help='When set to an integer value, QM9 will only contain molecules of that amount of atoms')
    parser.add_argument('--charge_power', type=int, default=2, metavar='N',
                        help='maximum power to take into one-hot features')
    parser.add_argument('--dataset', type=str, default="qm9_first_half", metavar='N',
                        help='qm9_first_half')
    parser.add_argument('--datadir', type=str, default="../../qm9/temp", metavar='N',
                        help='qm9_first_half')
    parser.add_argument('--remove_h', action='store_true')
    parser.add_argument('--include_charges', type=eval, default=True, help='include atom charge or not')
    parser.add_argument('--node_attr', type=int, default=0, metavar='N',
                        help='node_attr or not')
    parser.add_argument('--weight_decay', type=float, default=1e-16, metavar='N',
                        help='weight decay')
    parser.add_argument('--save_path', type=float, default=1e-16, metavar='N',
                        help='weight decay')
    parser.add_argument('--model_name', type=str, default='frad', metavar='N',
                        help='egnn | naive | numnodes')
    parser.add_argument('--save_model', type=eval, default=True)

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    dtype = torch.float32
    args.device = device
    print(args)
    
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    res = {'epochs': [], 'losess': [], 'best_val': 1e10, 'best_test': 1e10, 'best_epoch': 0}

    prop_utils.makedir(args.outf)
    prop_utils.makedir(args.outf + "/" + args.exp_name)
    args.world_size = 1
    args.rank = 0
    dataloaders, charge_scale = dataset.retrieve_dataloaders(args)
    args.dataset = "qm9_second_half"
    dataloaders_aux, _ = dataset.retrieve_dataloaders(args)
    dataloaders["test"] = dataloaders_aux["train"]

    # compute mean and mean absolute deviation
    property_norms = utils.compute_mean_mad_from_dataloader(dataloaders['valid'], [args.property])
    mean, mad = property_norms[args.property]['mean'], property_norms[args.property]['mad']

    model = get_model(args)

    print(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    for epoch in range(0, args.epochs):
        train(model, epoch, dataloaders['train'], mean, mad, args.property, device, partition='train', optimizer=optimizer, lr_scheduler=lr_scheduler, log_interval=args.log_interval)
        if epoch % args.test_interval == 0:
            val_loss = train(model, epoch, dataloaders['valid'], mean, mad, args.property, device, partition='valid', optimizer=optimizer, lr_scheduler=lr_scheduler, log_interval=args.log_interval)
            test_loss = test(model, epoch, dataloaders['test'], mean, mad, args.property, device, log_interval=args.log_interval)
            res['epochs'].append(epoch)
            res['losess'].append(test_loss)

            if val_loss < res['best_val']:
                res['best_val'] = val_loss
                res['best_test'] = test_loss
                res['best_epoch'] = epoch
                if args.save_model:
                    torch.save(model.state_dict(), args.outf + "/" + args.exp_name + "/best_checkpoint.npy")
                    with open(args.outf + "/" + args.exp_name + "/args.pickle", 'wb') as f:
                        pickle.dump(args, f)
            print("Val loss: %.4f \t test loss: %.4f \t epoch %d" % (val_loss, test_loss, epoch))
            print("Best: val loss: %.4f \t test loss: %.4f \t epoch %d" % (res['best_val'], res['best_test'], res['best_epoch']))

        json_object = json.dumps(res, indent=4)
        with open(args.outf + "/" + args.exp_name + "/losess.json", "w") as outfile:
            outfile.write(json_object)

