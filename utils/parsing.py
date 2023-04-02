from argparse import ArgumentParser
import subprocess, time
from .logging import get_logger
logger = get_logger(__name__)

def parse_train_args():
    
    parser = ArgumentParser()
    
    ## General args
    parser.add_argument('--workdir', type=str, default='./workdir', help='Model checkpoint root directory')
    parser.add_argument('--pdb_dir', type=str, default='./data/pdb_chains', help='Path to unpacked PDB chains')
    parser.add_argument('--dry_run', action='store_true', default=False)
    parser.add_argument('--splits', type=str, required=True, help='Path to splits CSV')
    parser.add_argument('--wandb', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=4, help='Dataloader workers')
    parser.add_argument('--resume', type=str, default=None, help='Path to model dir to continue training')
    parser.add_argument('--data_skip', action='store_true', default=True)
    parser.add_argument('--inference_mode', action='store_true', default=False)
    
    ### Preprocess args
    parser.add_argument('--sde_weight', type=float, default=1)
    parser.add_argument('--train_skew', type=float, default=0)
    parser.add_argument('--train_Hf', type=float, default=2)
    parser.add_argument('--train_kmin', type=int, default=5)
    parser.add_argument('--train_cutoff', type=float, default=5)
    parser.add_argument('--train_tmin', type=float, default=0.01)
    parser.add_argument('--train_rmsd_max', type=float, default=0.)
    
    ## Inference params
    parser.add_argument('--inf_type', type=str,
                        choices=['entropy', 'rate'], default='rate')
    parser.add_argument('--inf_step', type=float, default=0.5)
    parser.add_argument('--inf_freq', type=int, default=1)
    parser.add_argument('--inf_mols', type=int, default=100)
    
    ## OmegaFold args
    parser.add_argument('--omegafold_num_recycling', type=int, default=4)
    parser.add_argument('--embeddings_dir', type=str, default='./data/embeddings')
    parser.add_argument('--embeddings_key', type=str, choices=['name', 'reference'], default='reference')
    
    ## Model args
    parser.add_argument('--sde_a', type=float, default=3/(3.8**2))
    parser.add_argument('--sde_b', type=float, default=0)
    parser.add_argument('--resi_conv_layers', type=int, default=6)
    parser.add_argument('--resi_ns', type=int, default=32)
    parser.add_argument('--resi_nv', type=int, default=4)
    parser.add_argument('--resi_ntps', type=int, default=16)
    parser.add_argument('--resi_ntpv', type=int, default=4)
    parser.add_argument('--resi_fc_dim', type=int, default=128)
    parser.add_argument('--resi_pos_emb_dim', type=int, default=16)
    
    parser.add_argument('--lin_nf', type=int, default=1)
    parser.add_argument('--lin_self', action='store_true', default=False)
    parser.add_argument('--attention', action='store_true', default=False)
    parser.add_argument('--sh_lmax', type=int, default=2)
    parser.add_argument('--order', type=int, choices=[1, 2], default=1)
    parser.add_argument('--t_emb_dim', type=int, default=32)
    parser.add_argument('--t_emb_type', type=str, choices=['sinusoidal', 'fourier'], default='sinusoidal')
    parser.add_argument('--radius_emb_type', type=str, choices=['sinusoidal', 'gaussian'], default='gaussian')
    parser.add_argument('--radius_emb_dim', type=int, default=50)
    parser.add_argument('--radius_emb_max', type=float, default=50)
    parser.add_argument('--tmin', type=float, default=0.001),
    parser.add_argument('--tmax', type=float, default=1e6)
    parser.add_argument('--no_radius_sqrt', action='store_true', default=False)
    parser.add_argument('--parity', action='store_true', default=True)
    
    #### Training args
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--limit_mols', type=int, default=None)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--lr_start', type=float, default=1)
    parser.add_argument('--lr_end', type=float, default=1)
    parser.add_argument('--warmup_dur', type=float, default=1e4) # 3hrs
    parser.add_argument('--constant_dur', type=float, default=1e5) # 30 hrs
    parser.add_argument('--decay_dur', type=float, default=5e5) # 150 hrs
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--max_len', type=int, default=1500)
    parser.add_argument('--cuda_diffuse', action='store_true', default=False)
    
    parser.add_argument('--lm_edge_dim', type=int, default=128)
    parser.add_argument('--lm_node_dim', type=int, default=256)
    parser.add_argument('--no_edge_embs', action='store_true', default=False)
    
    args = parser.parse_args()
    args.time = int(time.time()*1000)
    args.commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    
    return args

