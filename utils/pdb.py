import torch, os, warnings, io, subprocess
from Bio.PDB import PDBIO, Chain, Residue, Polypeptide, Atom, PDBParser
from Bio import pairwise2
import numpy as np
from Bio.PDB.PDBExceptions import PDBConstructionWarning
parser = PDBParser()
from .logging import get_logger
logger = get_logger(__name__)
from .protein_residues import normal as RESIDUES
from Bio.SeqUtils import seq1, seq3

def PROCESS_RESIDUES(d):
    d['HIS'] = d['HIP']
    d = {key: val for key, val in d.items() if seq1(key) != 'X'}
    for key in d:
        atoms = d[key]['atoms']
        d[key] = {'CA': 'C'} | {key: val['symbol'] for key, val in atoms.items() if val['symbol'] != 'H' and key != 'CA'}
    return d
    
RESIDUES = PROCESS_RESIDUES(RESIDUES)

my_dir = f"/tmp/{os.getpid()}"
if not os.path.isdir(my_dir):
    os.mkdir(my_dir)

def pdb_to_npy(pdb_path, model_num=0, chain_id=None, seqres=None):
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PDBConstructionWarning)
        try:
            model = parser.get_structure('', pdb_path)[model_num]
            if chain_id is not None:
                chain = model.child_dict[chain_id]
            else:
                chain = model.child_list[0]
        except: logger.warning(f'Error opening PDB file {pdb_path}'); return

    coords = []
    seq = ''
    try:
        for resi in list(chain):
            if (resi.id[0] != ' ') or ('CA' not in resi.child_dict) or (resi.resname not in RESIDUES): continue
            co = np.zeros((14, 3)) * np.nan
            atoms = RESIDUES[resi.resname]
            seq += resi.resname
            for i, at in enumerate(atoms):
                try: co[i] = resi.child_dict[at].coord
                except: pass
            coords.append(co)
        coords = np.stack(coords)
        seq = seq1(seq)
    except: logger.warning(f'Error parsing chain {pdb_path}'); return
    if not seqres: return coords, seq

    coords_= np.zeros((len(seqres), 14, 3)) * np.nan
    alignment = pairwise2.align.globalxx(seq, seqres)[0]
    
    if '-' in alignment.seqB: 
        logger.warning(f'Alignment gaps {pdb_path}'); return
    mask = np.array([a == b for a, b in zip(alignment.seqA, alignment.seqB)])
    coords_[mask] = coords
    return coords_, mask

def tmscore(X_path, Y_path, molseq=None, lddt=True, lddt_start=1):
    if type(X_path) is not str:
        PDBFile(molseq).add(X_path).write(os.path.join(my_dir, 'X.pdb'))
        X_path = os.path.join(my_dir, 'X.pdb')
    if type(Y_path) is not str:
        PDBFile(molseq).add(Y_path).write(os.path.join(my_dir, 'Y.pdb'))
        Y_path = os.path.join(my_dir, 'Y.pdb')
    
    if not os.path.isabs(X_path): X_path = os.path.join(os.getcwd(), X_path)
    if not os.path.isabs(Y_path): Y_path = os.path.join(os.getcwd(), Y_path)
    
    out = subprocess.check_output(['TMscore', '-seq', Y_path, X_path], 
                    stderr=open('/dev/null', 'w'), cwd=my_dir)
    start = out.find(b'RMSD')
    end = out.find(b'rotation')
    out = out[start:end]
    
    rmsd, _, tm, _, gdt_ts, gdt_ha, _, _ = out.split(b'\n')
    
    rmsd = float(rmsd.split(b'=')[-1])
    tm = float(tm.split(b'=')[1].split()[0])
    gdt_ts = float(gdt_ts.split(b'=')[1].split()[0])
    gdt_ha = float(gdt_ha.split(b'=')[1].split()[0])
    
    if lddt:
        X_renum = os.path.join(my_dir, 'X_renum.pdb')
        renumber_pdb(molseq, X_path, X_renum, start=lddt_start)
        out = subprocess.check_output(['lddt', '-c', Y_path, X_renum],  # reference comes last
            stderr=open('/dev/null', 'w'), cwd=my_dir)
        for line in out.split(b'\n'):
            if b'Global LDDT score' in line:
                lddt = float(line.split(b':')[-1].strip())
        return {'rmsd': rmsd, 'tm': tm, 'gdt_ts': gdt_ts, 'gdt_ha': gdt_ha, 'lddt': lddt}
    else:
        return {'rmsd': rmsd, 'tm': tm, 'gdt_ts': gdt_ts, 'gdt_ha': gdt_ha}

    
class PDBFile:
    def __init__(self, molseq):
        self.molseq = molseq
        self.blocks = []
        self.chain = chain = Chain.Chain('A')
        j = 1
        for i, aa in enumerate(molseq):
            aa = Residue.Residue(
                id=(' ', i+1, ' '), 
                resname=seq3(aa).upper(), 
                segid='    '
            )
            for atom in RESIDUES[aa.resname.upper()]:
                at = Atom.Atom(
                    name=atom, coord=None, bfactor=0, occupancy=1, altloc=' ',
                    fullname=f' {atom} ', serial_number=j, element=RESIDUES[aa.resname][atom]
                )
                j += 1
                aa.add(at)
            chain.add(aa)
       
          
    def add(self, coords):
        if type(coords) is np.ndarray:
            coords = coords.astype(np.float64)
        elif type(coords) is torch.Tensor:
            coords = coords.cpu().double().numpy()
        
        for i, resi in enumerate(self.chain):
            atoms = RESIDUES[resi.resname]
            resi['CA'].coord = coords[i]

        k = len(self.chain)
        for i, resi in enumerate(self.chain):
            atoms = RESIDUES[resi.resname]
            for j, at in enumerate(atoms):
                if at != 'CA': 
                    try: resi[at].coord = coords[k] + resi['CA'].coord
                    except: resi[at].coord = (np.nan, np.nan, np.nan)
                    k+=1

        pdbio = PDBIO()
        pdbio.set_structure(self.chain)

        stringio = io.StringIO()
        stringio.close, close = lambda: None, stringio.close
        pdbio.save(stringio)
        block = stringio.getvalue().split('\n')[:-2]
        stringio.close = close; stringio.close()

        self.blocks.append(block)
        return self
        
    def clear(self):
        self.blocks = []
        return self
    
    def write(self, path=None, idx=None, reverse=False):
        is_first = True
        str_ = ''
        blocks = self.blocks[::-1] if reverse else self.blocks
        for block in ([blocks[idx]] if idx is not None else blocks):
            block = [line for line in block if 'nan' not in line]
            if not is_first:
                block = [line for line in block if 'CONECT' not in line]
            else:
                is_first = False
            str_ += 'MODEL\n'
            str_ += '\n'.join(block)
            str_ += '\nENDMDL\n'
        if not path:
            return str_
        with open(path, 'w') as f:
            f.write(str_)
            
def renumber_pdb(molseq, X_path, X_renum, start=1):
    assert type(molseq) == str
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PDBConstructionWarning)
        model = parser.get_structure('', X_path)[0]
    chain = model.child_list[0]
    polypeptide = Polypeptide.Polypeptide(chain.get_residues())
    seq = polypeptide.get_sequence()
    
    alignment = pairwise2.align.globalxx(seq, molseq)[0]
    assert '-' not in alignment.seqB
    numbering = [i+start for i, c in enumerate(alignment.seqA) if c != '-']
    
    for n, resi in enumerate(chain):
        resi.id = (' ', 10000+n, ' ')
    
    for n, resi in zip(numbering, chain):
        resi.id = (' ', n, ' ')
        
    pdbio = PDBIO()
    pdbio.set_structure(chain)
    pdbio.save(X_renum)