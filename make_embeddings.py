import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--splits', type=str, required=True)
parser.add_argument('--out_dir', type=str, default='./data/embeddings')
parser.add_argument('--device', type=str, default="cuda")
parser.add_argument('--lm_weights_path', default="release1.pt")
parser.add_argument('--omegafold_num_recycling', type=int, default=4)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--worker_id', type=int, default=0)
parser.add_argument('--reference_only', action='store_true', default=False)
args, _ = parser.parse_known_args()

import pandas as pd
import numpy as np
import tqdm, os, torch
from omegafold.__main__ import OmegaFoldModel

def main():
    """
    Featurizes amino acids into node and edge embeddings.
    Embeddings are stored as a dict: {"node_repr": <EDGE_REPR>, "edge_repr": <EDGE_REPR>} 
    """
    
    splits = pd.read_csv(args.splits).set_index("name").sort_values('seqlen')
    
    if args.reference_only:
        splits = splits[(splits.index == splits.reference)]
    
    splits = splits.iloc[args.worker_id::args.num_workers]
    
    arg_keys = ['omegafold_num_recycling']
    suffix = get_args_suffix(arg_keys, args) + '.npz'

    # load OmegaFold model
    omegafold = OmegaFoldModel(args.lm_weights_path, device=args.device)
    skipping = 0
    doing = 0
    for path in tqdm.tqdm(splits.index):
        embeddings_dir = os.path.join(args.out_dir, path[:2])
        if not os.path.exists(embeddings_dir): os.makedirs(embeddings_dir)
        embeddings_path = os.path.join(embeddings_dir, path)  + '.' + suffix
        
        if os.path.exists(embeddings_path): 
            skipping += 1
            continue
        
        doing += 1
        fasta_lines = [f">{path}", splits.loc[path]["seqres"]]

        try:
            edge_results, node_results = omegafold.inference(
                fasta_lines, args.omegafold_num_recycling
            )

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f'CUDA OOM, skipping {path}')
                torch.cuda.empty_cache()
                continue
            
            else:
                logger.error("Uncaught error")
                raise e
        np.savez(embeddings_path, node_repr=node_results[0], edge_repr=edge_results[0])
    print(args.splits, 'DONE')
    print('Skipped', skipping)
    print('Done', doing)
    
def get_args_suffix(arg_keys, args):
    cache_name = []
    for k in arg_keys: cache_name.extend([k, args.__dict__[k]])
    return '.'.join(map(str, cache_name))

if __name__ == '__main__':
    main()