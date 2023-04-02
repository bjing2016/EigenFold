import copy, torch, os
from diffusion.sampling import reverse_sample, logp
from diffusion import schedule
import numpy as np
from .pdb import PDBFile, tmscore
from .logging import get_logger

logger = get_logger(__name__)

def inference_epoch(args, model, dataset, device='cpu', rank=0, world_size=1, pdbs=False, elbo=None):
    model.eval()
    samples = []
    N = min(len(dataset), args.inf_mols)
    num_samples = args.num_samples
    datas = []
    for i in range(rank, N, world_size):
        
        data_ = dataset.get(i); sde = data_.sde
        if data_.skip:
            logger.warning('Skipping inference mol'); continue    
        molseq = data_.info.seqres
        sched = get_schedule(args, data_.resi_sde)
        sched_full = get_schedule(args, data_.resi_sde, full=True)
        score_fn = get_score_fn(args, model, data_, key='resi', device=device)
        
        for j in range(num_samples):        
            try:
                pdb = PDBFile(molseq) if pdbs else None
                data = copy.deepcopy(data_)
                data.Y = reverse_sample(args, score_fn, sde, sched, device=device, Y=None,
                            pdb=pdb, tqdm_=not args.wandb, ode=args.ode)
                
                data.elbo_Y = logp(data.Y, score_fn, sde, sched_full, device=device, tqdm_=False) if elbo else np.nan
                
                if pdb: data.pdb = pdb
                if os.path.exists(data.path):
                    res = tmscore(data.path, data.Y, molseq)
                else:
                    res = {'rmsd': np.nan, 'gdt_ts': np.nan, 'gdt_ha': np.nan, 'tm': np.nan, 'lddt': np.nan}
                
                data.__dict__.update(res)
                data.copy = j; datas.append(data)
                logger.info(f'{data.path} ELBO_Y {data.elbo_Y} {res}')
                
            except Exception as e:
                if type(e) is KeyboardInterrupt: raise e
                logger.error('Skipping inference mol due to exception ' + str(e))
                raise e
                
    log = {
        'path': [data.path for data in datas],
        'copy': [data.copy for data in datas],
        'elbo_Y': [data.elbo_Y for data in datas],
        'rmsd': [data.rmsd for data in datas],
        'gdt_ts': [data.gdt_ts for data in datas],
        'gdt_ha': [data.gdt_ha for data in datas],
        'tm': [data.tm for data in datas],
        'lddt': [data.lddt for data in datas]
    }
        
    return datas, log

def get_score_fn(args, model, data, key='resi', device='cpu'):
    data = copy.deepcopy(data); data.to(device); sde = data.sde
    @torch.no_grad()
    def score_fn(Y, t, k):
        data[key].pos = Y[:data['resi'].num_nodes]
        data[key].node_t = torch.ones(data.resi_sde.N, device=device) * t
        data.score_norm = sde.score_norm(t, k, adj=True)
        data['sidechain'].pos = Y[data['resi'].num_nodes:]
        return model.enn(data)
    return score_fn

def get_schedule(args, sde, full=False):
    return {
        'entropy': schedule.EntropySchedule,
        'rate': schedule.RateSchedule
    }[args.inf_type](
        sde,
        Hf=args.train_Hf,
        rmsd_max=0,
        step=args.elbo_step if full else args.inf_step,
        cutoff=args.train_cutoff,
        kmin=args.train_kmin,
        tmin=args.train_tmin,
        alpha=0 if full else args.alpha,
        beta=1 if full else args.beta
    )