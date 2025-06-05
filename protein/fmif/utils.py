
import torch
from torch.utils.data import DataLoader
import csv
from dateutil import parser
import numpy as np
import time
import random
import os
from tqdm import tqdm


class StructureDataset():
    def __init__(self, pdb_dict_list, verbose=True, truncate=None, max_length=100,
                 alphabet='ACDEFGHIKLMNPQRSTVWYX'):
        alphabet_set = set([a for a in alphabet])
        discard_count = {
            'bad_chars': 0,
            'too_long': 0,
            'bad_seq_length': 0
        }

        self.data = []

        start = time.time()
        for i, entry in enumerate(pdb_dict_list):
            seq = entry['seq']
            name = entry['name']  # pdbid_cid

            bad_chars = set([s for s in seq]).difference(alphabet_set)
            if len(bad_chars) == 0:
                if len(entry['seq']) <= max_length:
                    self.data.append(entry)
                else:
                    discard_count['too_long'] += 1
            else:
                discard_count['bad_chars'] += 1

            # Truncate early
            if truncate is not None and len(self.data) == truncate:
                return

            if verbose and (i + 1) % 1000 == 0:
                elapsed = time.time() - start
                # print('{} entries ({} loaded) in {:.1f} s'.format(len(self.data), i+1, elapsed))

            # print('Discarded', discard_count)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class StructureLoader():
    def __init__(self, dataset, batch_size=100, shuffle=True,
                 collate_fn=lambda x: x, drop_last=False):
        self.dataset = dataset
        self.size = len(dataset)
        self.lengths = [len(dataset[i]['seq']) for i in range(self.size)]
        self.batch_size = batch_size
        sorted_ix = np.argsort(self.lengths)

        # Cluster into batches of similar sizes
        clusters, batch = [], []
        batch_max = 0
        for ix in sorted_ix:
            size = self.lengths[ix]
            if size * (len(batch) + 1) <= self.batch_size:
                batch.append(ix)
                batch_max = size
            else:
                clusters.append(batch)
                batch, batch_max = [], 0
        if len(batch) > 0:
            clusters.append(batch)
        self.clusters = clusters

    def __len__(self):
        return len(self.clusters)

    def __iter__(self):
        np.random.shuffle(self.clusters)
        for b_idx in self.clusters:
            batch = [self.dataset[i] for i in b_idx]
            yield batch


def worker_init_fn(worker_id):
    np.random.seed()


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


def get_pdbs(data_loader, repeat=1, max_length=10000, num_units=1000000, load_repr=True):
    init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                     'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                     'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    extra_alphabet = [str(item) for item in list(np.arange(300))]
    chain_alphabet = init_alphabet + extra_alphabet
    c = 0
    c1 = 0
    pdb_dict_list = []
    t0 = time.time()
    from tqdm import tqdm
    for _ in range(repeat):
        for step, t in tqdm(enumerate(data_loader)):
            t = {k: v[0] for k, v in t.items()}
            c1 += 1
            if 'label' in list(t):
                my_dict = {}
                s = 0
                concat_seq = ''
                concat_N = []
                concat_CA = []
                concat_C = []
                concat_O = []
                concat_mask = []
                coords_dict = {}
                mask_list = []
                visible_list = []
                if len(list(np.unique(t['idx']))) < 352:
                    # assert len(np.unique(t['idx'])) == 1, 'only support single chain'
                    for idx in list(np.unique(t['idx'])):
                        letter = chain_alphabet[idx]
                        res = np.argwhere(t['idx'] == idx)  # [1, L]
                        # print(res.shape, t['idx'].shape)
                        initial_sequence = "".join(list(np.array(list(t['seq']))[res][0,]))
                        if initial_sequence[-6:] == "HHHHHH":
                            res = res[:, :-6]
                        if initial_sequence[0:6] == "HHHHHH":
                            res = res[:, 6:]
                        if initial_sequence[-7:-1] == "HHHHHH":
                            res = res[:, :-7]
                        if initial_sequence[-8:-2] == "HHHHHH":
                            res = res[:, :-8]
                        if initial_sequence[-9:-3] == "HHHHHH":
                            res = res[:, :-9]
                        if initial_sequence[-10:-4] == "HHHHHH":
                            res = res[:, :-10]
                        if initial_sequence[1:7] == "HHHHHH":
                            res = res[:, 7:]
                        if initial_sequence[2:8] == "HHHHHH":
                            res = res[:, 8:]
                        if initial_sequence[3:9] == "HHHHHH":
                            res = res[:, 9:]
                        if initial_sequence[4:10] == "HHHHHH":
                            res = res[:, 10:]
                        if res.shape[1] < 4:
                            pass
                        else:
                            my_dict['seq_chain_' + letter] = "".join(list(np.array(list(t['seq']))[res][0,]))
                            concat_seq += my_dict['seq_chain_' + letter]
                            if idx in t['masked']:
                                mask_list.append(letter)
                            else:
                                visible_list.append(letter)
                            coords_dict_chain = {}
                            all_atoms = np.array(t['xyz'][res,])[0,]  # [L, 14, 3]
                            coords_dict_chain['N_chain_' + letter] = all_atoms[:, 0, :].tolist()
                            coords_dict_chain['CA_chain_' + letter] = all_atoms[:, 1, :].tolist()
                            coords_dict_chain['C_chain_' + letter] = all_atoms[:, 2, :].tolist()
                            coords_dict_chain['O_chain_' + letter] = all_atoms[:, 3, :].tolist()
                            my_dict['coords_chain_' + letter] = coords_dict_chain
                            if load_repr:
                                my_dict['repr_single_' + letter] = t['repr_single'][res,][0,].to(torch.float16)
                                my_dict['repr_pair_' + letter] = t['repr_pair'][res,][0,][:, res, ][:, 0, ].to(torch.float16)
                                my_dict['repr_structure_' + letter] = (t['repr_structure'][res,][0,]*100.0).to(torch.float16)

                    my_dict['name'] = t['label']
                    my_dict['masked_list'] = mask_list
                    my_dict['visible_list'] = visible_list
                    my_dict['num_of_chains'] = len(mask_list) + len(visible_list)
                    my_dict['seq'] = concat_seq
                    if len(concat_seq) <= max_length:
                        pdb_dict_list.append(my_dict)
                    if len(pdb_dict_list) >= num_units:
                        # raise ValueError('Number of units exceeded')
                        break
    return pdb_dict_list


class PDB_dataset(torch.utils.data.Dataset):
    def __init__(self, IDs, loader, train_dict, params, load_repr=True):
        self.IDs = IDs
        self.train_dict = train_dict
        self.loader = loader
        self.params = params
        self.load_repr = load_repr

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index):
        ID = self.IDs[index]
        # sel_idx = np.random.randint(0, len(self.train_dict[ID]))
        sel_idx = 0  # always select the first one from all sequences in this cluster
        out = self.loader(self.train_dict[ID][sel_idx], self.params, ID, load_repr=self.load_repr)
        return out


class PDB_dataset_withclusterid(torch.utils.data.Dataset):
    def __init__(self, IDs, loader, train_dict, params):
        self.IDs = IDs
        self.train_dict = train_dict
        self.loader = loader
        self.params = params

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index):
        ID = self.IDs[index]
        # sel_idx = np.random.randint(0, len(self.train_dict[ID]))
        sel_idx = 0  # always select the first one from all sequences in this cluster
        out = self.loader(self.train_dict[ID][sel_idx], self.params, ID)
        return out, ID


def loader_pdb(item, params, cluster_id, load_repr=True):
    pdbid, chid = item[0].split('_')
    PREFIX = "%s/pdb/%s/%s" % (params['DIR'], pdbid[1:3], pdbid)

    # load metadata
    if not os.path.isfile(PREFIX + ".pt"):
        return {'seq': np.zeros(5)}
    meta = torch.load(PREFIX + ".pt")
    asmb_ids = meta['asmb_ids']
    asmb_chains = meta['asmb_chains']
    chids = np.array(meta['chains'])

    # find candidate assemblies which contain chid chain
    # asmb_candidates = set([a for a,b in zip(asmb_ids,asmb_chains)
    #                        if chid in b.split(',')])
    asmb_candidates = set([])
    # if the chains is missing is missing from all the assemblies
    # then return this chain alone
    if len(asmb_candidates) < 1:
        chain = torch.load("%s_%s.pt" % (PREFIX, chid))
        L = len(chain['seq'])
        if load_repr:
            reprs = np.load(f"{params['REPR_DIR']}/cluster_{cluster_id}/seed-0_embeddings/embeddings.npz")
            return {
                'seq': chain['seq'],
                'xyz': chain['xyz'],
                'idx': torch.zeros(L).int(),
                'masked': torch.Tensor([0]).int(),
                'label': item[0],
                'repr_single': reprs['single_embeddings'][:L, :],  # 384
                'repr_pair': reprs['pair_embeddings'][:L, :L, :],  # 128
                'repr_structure': reprs['structure_embeddings'][:L, :],  # 768
            }
        else:
            return {
                'seq': chain['seq'],
                'xyz': chain['xyz'],
                'idx': torch.zeros(L).int(),
                'masked': torch.Tensor([0]).int(),
                'label': item[0],
            }

    # randomly pick one assembly from candidates
    asmb_i = random.sample(list(asmb_candidates), 1)

    # indices of selected transforms
    idx = np.where(np.array(asmb_ids) == asmb_i)[0]

    # load relevant chains
    chains = {c: torch.load("%s_%s.pt" % (PREFIX, c))
              for i in idx for c in asmb_chains[i]
              if c in meta['chains']}

    # generate assembly
    asmb = {}
    for k in idx:

        # pick k-th xform
        xform = meta['asmb_xform%d' % k]
        u = xform[:, :3, :3]
        r = xform[:, :3, 3]

        # select chains which k-th xform should be applied to
        s1 = set(meta['chains'])
        s2 = set(asmb_chains[k].split(','))
        chains_k = s1 & s2

        # transform selected chains
        for c in chains_k:
            try:
                xyz = chains[c]['xyz']
                xyz_ru = torch.einsum('bij,raj->brai', u, xyz) + r[:, None, None, :]
                asmb.update({(c, k, i): xyz_i for i, xyz_i in enumerate(xyz_ru)})
            except KeyError:
                return {'seq': np.zeros(5)}

    # select chains which share considerable similarity to chid
    seqid = meta['tm'][chids == chid][0, :, 1]
    homo = set([ch_j for seqid_j, ch_j in zip(seqid, chids)
                if seqid_j > params['HOMO']])
    # stack all chains in the assembly together
    seq, xyz, idx, masked = "", [], [], []
    seq_list = []
    for counter, (k, v) in enumerate(asmb.items()):
        seq += chains[k[0]]['seq']
        seq_list.append(chains[k[0]]['seq'])
        xyz.append(v)
        idx.append(torch.full((v.shape[0],), counter))
        if k[0] in homo:
            masked.append(counter)

    # print(len(seq), torch.cat(xyz,dim=0).shape, torch.cat(idx,dim=0).shape)

    return {'seq': seq,
            'xyz': torch.cat(xyz, dim=0),
            'idx': torch.cat(idx, dim=0),
            'masked': torch.Tensor(masked).int(),
            'label': item[0]}


def build_training_clusters(params, debug):
    val_ids = set([int(l) for l in open(params['VAL']).readlines()])
    test_ids = set([int(l) for l in open(params['TEST']).readlines()])

    if debug:
        val_ids = []
        test_ids = []

    # read & clean list.csv
    with open(params['LIST'], 'r') as f:
        reader = csv.reader(f)
        next(reader)
        rows = [[r[0], r[3], int(r[4])] for r in reader
                if float(r[2]) <= params['RESCUT'] and
                parser.parse(r[1]) <= parser.parse(params['DATCUT']) and len(r[-1]) <= params['MAXLEN']]


    # compile training and validation sets
    train = {}
    valid = {}
    test = {}

    if debug:
        rows = rows[:20]
    for r in rows:
        # if len(r[-1]) > params['MAXLEN']:
        #     continue
        if r[2] in val_ids:
            if r[2] in valid.keys():
                valid[r[2]].append(r[:2])
            else:
                valid[r[2]] = [r[:2]]
        elif r[2] in test_ids:
            if r[2] in test.keys():
                test[r[2]].append(r[:2])
            else:
                test[r[2]] = [r[:2]]
        else:
            if r[2] in train.keys():
                train[r[2]].append(r[:2])
            else:
                train[r[2]] = [r[:2]]
    if debug:
        valid = train
        test = train
        # key: cluster id; value: list of chain_id and hash
    return train, valid, test


def build_whole_clusters(params, debug):
    # read & clean list.csv
    with open(params['LIST'], 'r') as f:
        reader = csv.reader(f)
        next(reader)
        rows = [[r[0], r[3], int(r[4])] for r in reader
                if float(r[2]) <= params['RESCUT'] and
                parser.parse(r[1]) <= parser.parse(params['DATCUT'])]

    data_dict = {}

    if debug:
        rows = rows[:20]
    for r in rows:
        if r[2] in data_dict.keys():
            data_dict[r[2]].append(r[:2])
        else:
            data_dict[r[2]] = [r[:2]]
    # key: cluster id; value: list of chain_id and hash
    return data_dict


def from_same_class_simple(s1: str, s2: str) -> bool:
    """
    Returns True if s1 and s2 are from the same class, where 'X' in either string
    can match either an empty string or exactly one arbitrary letter, and
    we want to see if there's *some* way to make s1 and s2 identical after
    those replacements.
    """

    # Memo dict for dp(i, j) -> bool
    memo = {}

    def dp(i, j):
        # If we've seen this state before, return the stored result
        if (i, j) in memo:
            return memo[(i, j)]

        # If both pointers are at the end, perfect match
        if i == len(s1) and j == len(s2):
            memo[(i, j)] = True
            return True

        # If s1 is exhausted but s2 is not...
        if i == len(s1):
            # The only way to match is if the remaining chars in s2 are all 'X' and used as empty
            if j < len(s2) and s2[j] == 'X':
                ans = dp(i, j + 1)  # skip this 'X'
            else:
                ans = False
            memo[(i, j)] = ans
            return ans

        # If s2 is exhausted but s1 is not...
        if j == len(s2):
            # The only way to match is if the remaining chars in s1 are all 'X' and used as empty
            if i < len(s1) and s1[i] == 'X':
                ans = dp(i + 1, j)
            else:
                ans = False
            memo[(i, j)] = ans
            return ans

        # Now both i < len(s1) and j < len(s2)
        c1, c2 = s1[i], s2[j]

        if c1 != 'X' and c2 != 'X':
            # Both are letters; must match
            ans = (c1 == c2) and dp(i + 1, j + 1)

        elif c1 == 'X' and c2 != 'X':
            # s1[i] can be empty or match s2[j]
            # 1) skip s1[i] => dp(i+1, j)
            # 2) match one letter => dp(i+1, j+1)
            ans = dp(i + 1, j) or dp(i + 1, j + 1)

        elif c1 != 'X' and c2 == 'X':
            # Similarly
            ans = dp(i, j + 1) or dp(i + 1, j + 1)

        else:
            # c1 == 'X' and c2 == 'X'
            # Both can be empty or the same letter => both lead to the same subproblem
            ans = dp(i + 1, j + 1)

        memo[(i, j)] = ans
        return ans

    return dp(0, 0)


def from_same_class(s1: str, s2: str) -> bool:
    """
    Checks if there's some string T such that s1 can become T and s2 can become T
    by replacing each 'X' with any (possibly empty) sequence of letters.
    """
    from functools import lru_cache

    @lru_cache(None)
    def dp(i, j):
        # If both are at end, perfect match
        if i == len(s1) and j == len(s2):
            return True

        # If s1 is exhausted, see if s2 has only 'X' left (which can be empty)...
        if i == len(s1):
            # We can only succeed if all remaining chars in s2 can be
            # turned into empty (i.e., they are 'X' and we skip them).
            if j < len(s2) and s2[j] == 'X':
                return dp(i, j + 1)
            return False

        # If s2 is exhausted, symmetrical check
        if j == len(s2):
            if i < len(s1) and s1[i] == 'X':
                return dp(i + 1, j)
            return False

        c1, c2 = s1[i], s2[j]

        # Case 1: Both are ordinary letters
        if c1 != 'X' and c2 != 'X':
            if c1 == c2:
                return dp(i + 1, j + 1)
            else:
                return False

        # Case 2: s1[i] == 'X' and s2[j] != 'X'
        if c1 == 'X' and c2 != 'X':
            # Option A: 'X' matches empty => move i
            if dp(i + 1, j):
                return True
            # Option B: 'X' in s1 matches one or more letters in s2
            # We'll keep matching letters in s2 until the next 'X' or end
            # and see if that helps.
            # Because as soon as we see s2[k] == 'X', that belongs to a
            # different placeholder, so we stop.
            pos = j
            while pos < len(s2) and s2[pos] != 'X':
                # match s2[j..pos] as one chunk => we consume that chunk on s2 side
                # but only move past 'X' in s1 by 1 (i.e. i+1).
                # So s1[i] => that entire chunk
                if dp(i + 1, pos + 1):
                    return True
                pos += 1
            return False

        # Case 3: s1[i] != 'X' and s2[j] == 'X'
        if c1 != 'X' and c2 == 'X':
            # Symmetrical logic
            # Option A: skip 'X' in s2
            if dp(i, j + 1):
                return True
            # Option B: 'X' in s2 matches one or more letters in s1
            pos = i
            while pos < len(s1) and s1[pos] != 'X':
                if dp(pos + 1, j + 1):
                    return True
                pos += 1
            return False

        # Case 4: Both are 'X'
        # They can both skip => (i+1, j+1)
        if dp(i + 1, j + 1):
            return True

        # Or they can match some *identical* nonempty substring from each side.
        # Let's find how far we can go in s1 before hitting another 'X',
        # and how far in s2 before hitting another 'X'.
        nextX1 = i + 1
        while nextX1 < len(s1) and s1[nextX1] != 'X':
            nextX1 += 1

        nextX2 = j + 1
        while nextX2 < len(s2) and s2[nextX2] != 'X':
            nextX2 += 1

        # Now s1[i+1 : nextX1] is a maximal block of letters
        # and s2[j+1 : nextX2] is a maximal block of letters
        # We can try *all* splits of that block, but for the substring expansions
        # to be the same, they must be the same length and identical letters.

        # For each possible length in that block of s1...
        for cut1 in range(i + 1, nextX1 + 1):
            # The substring s1[i+1 : cut1]
            block1 = s1[i + 1:cut1]
            length_block1 = len(block1)

            # We can attempt to match the same length in s2's block
            # from j+1 onward, in increments up to nextX2.
            cut2 = j + 1 + length_block1
            if cut2 <= nextX2:  # make sure we don't overshoot
                block2 = s2[j + 1:cut2]
                # If they match exactly, then we can recurse from (cut1, cut2).
                if block1 == block2:
                    if dp(cut1, cut2):
                        return True
            else:
                # If we ran out of letters in s2's block, no point continuing
                break

        return False

    return dp(0, 0)


def build_total_seq_clusters(params, debug):
    # val_ids = set([int(l) for l in open(params['VAL']).readlines()])
    # test_ids = set([int(l) for l in open(params['TEST']).readlines()])

    # if debug:
    #     val_ids = []
    #     test_ids = []
    # print(from_same_class("XaXb", "XabX"))

    # read & clean list.csv
    with open(params['LIST'], 'r') as f:
        reader = csv.reader(f)
        next(reader)
        rows = [[r[0], r[3], int(r[4]), r[5]] for r in reader]  # pdbid_chainid, hash, cluster, sequence
        # if float(r[2])<=params['RESCUT'] and
        # parser.parse(r[1])<=parser.parse(params['DATCUT'])]

    # compile training and validation sets
    # train = {}
    # valid = {}
    # test = {}
    data_dict = {}
    seq_list = []

    if debug:
        rows = rows[:20000]
    for r in tqdm(rows):
        # if r[2] in val_ids:
        #     if r[2] in valid.keys():
        #         valid[r[2]].append(r[:2])
        #     else:
        #         valid[r[2]] = [r[:2]]
        # elif r[2] in test_ids:
        #     if r[2] in test.keys():
        #         test[r[2]].append(r[:2])
        #     else:
        #         test[r[2]] = [r[:2]]
        # else:

        # check that the SEQUENCE item corresponds to the seq in pdbid_chainid.pt
        pdbid, chid = r[0].split('_')
        PREFIX = "%s/pdb/%s/%s" % (params['DIR'], pdbid[1:3], pdbid)
        chain = torch.load("%s_%s.pt" % (PREFIX, chid))
        seq_list.append(chain['seq'])
        continue

        if len(r[3]) != len(chain['seq']):
            try:
                same_check = from_same_class_simple(r[3], chain['seq'])
            except:
                print(f"Error in from_same_class_simple of {pdbid}_{chid}.pt")
                same_check = False
            if not same_check:
                try:
                    same_check = from_same_class(r[3], chain['seq'])
                except:
                    print(f"Error in from_same_class of {pdbid}_{chid}.pt")
                    same_check = False
                if not same_check:
                    print(f"SEQUENCE item does not correspond to the seq in {pdbid}_{chid}.pt")
            # print(f"length of SEQUENCE item does not correspond to the seq in {pdbid}_{chid}.pt")
        else:
            # 1 to 1 comparison of SEQUENCE item and seq in pdbid_chainid.pt
            countdiff = sum(c1 != c2 for c1, c2 in zip(r[3], chain['seq']) if c1 != 'X')
            if countdiff > 1:
                print(f"non X differences in {pdbid}_{chid}.pt")

        if r[1] in data_dict.keys():
            data_dict[r[1]].append([r[0]] + r[2:] + [chain['seq']])
        else:
            data_dict[r[1]] = [[r[0]] + r[2:] + [chain['seq']]]
    # if debug:
    #     valid=train
    #     test=train
    # key: cluster id; value: list of chain_id and hash
    return data_dict, seq_list


def set_seed(seed, use_cuda):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    if use_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f'=> Seed of the run set to {seed}')
