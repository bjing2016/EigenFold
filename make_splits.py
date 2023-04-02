import pandas as pd
import os, json, tqdm, random, subprocess
from collections import defaultdict
from functools import partial
import numpy as np
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

def train_splits():
    df = pd.read_csv('data/pdb_chains.csv', index_col='name')
    df['seqlen'] = [len(s) for s in df.seqres]
    df = df[df.valid_alphas > 0]
    df = df[df.saved]; del df['saved']
    df = df[(df.seqlen >= 20) & (df.seqlen <= 256)]
    df = df[df.release_date < '2020-12-01']
    df['split'] = np.where(df.release_date < '2020-05-01', 'train', 'val')
    df.to_csv('splits/limit256.csv')
    
def apo_splits():
    df = pd.read_csv('data/pdb_chains.csv', index_col='name')
    df['seqlen'] = [len(s) for s in df.seqres]
    names = []
    # from https://gitlab.com/sbgunq/publications/af2confdiv-oct2021/-/blob/main/data/26-01-2022/revision1_86_plus_5.csv
    apo = pd.read_csv('splits/revision1_86_plus_5.csv', sep=';') 
    for _, row in apo.iterrows():
        name1 = row.apo_id[:4].lower() + '.' + row.apo_id[-1] + '.pdb'
        name2 = row.holo_id[:4].lower() + '.' + row.holo_id[-1] + '.pdb'
        if name1 not in df.index or name2 not in df.index: continue
        names.append(name1); names.append(name2)
    
    df = df.loc[names]  
    names = []
    others = []
    for i in range(0, len(df), 2):
        if 'X' in df.seqres[i]: continue
        names.append(df.index[i]); others.append(df.index[i+1])
    
    df = df.loc[names]
    df['holo'] = others
    del df['reference']; del df['saved']
    df = df[df.seqlen <= 750] 
    df.to_csv('splits/apo.csv')

    
def codnas_splits():
    df = pd.read_csv('data/pdb_chains.csv', index_col='name')
    df['seqlen'] = [len(s) for s in df.seqres]
    names = []
    # from https://onlinelibrary.wiley.com/action/downloadSupplement?doi=10.1002%2Fpro.4353&file=pro4353-sup-0002-TableS1.zip
    codnas = pd.read_csv('splits/codnas_orig.csv') 
    for _, row in codnas.iterrows():
        name1 = row.Fold1[:4] + '.' + row.Fold1[-1] + '.pdb'
        name2 = row.Fold2[:4] + '.' + row.Fold2[-1] + '.pdb'
        if name1 not in df.index or name2 not in df.index: continue
        names.append(name1); names.append(name2)
    
    df = df.loc[names]
    names = []
    others = []
    for i in range(0, len(df), 2):
        maxlen = max(df.seqlen[i:i+2])
        minlen = min(df.seqlen[i:i+2])
        if maxlen / minlen > 1.5: continue
        if df.seqlen[i+1] < df.seqlen[i]:
            names.append(df.index[i+1]); others.append(df.index[i])
        else:
            names.append(df.index[i]); others.append(df.index[i+1])
    
    df = df.loc[names]
    df['other'] = others
    del df['reference']; del df['saved']
    df = df[df.seqlen <= 750] 
    df.to_csv('splits/codnas.csv')
                         
def cameo_splits():   
    df = pd.read_csv('data/pdb_chains.csv', index_col='name')
    df['seqlen'] = [len(s) for s in df.seqres]
    cameo = pd.read_csv('splits/cameo2022_orig.csv') # from https://www.cameo3d.org/modeling/targets/1-year/?to_date=2022-12-31
    cameo['name'] = [s.replace('[', '').replace(']', '').replace(' ', '.') + '.pdb' for s in cameo['ref. PDB [Chain]']]
    cameo = cameo.set_index('name').join(df[['release_date', 'seqres']], how='inner')
    
    tosave = cameo[(cameo.release_date < '2022-11-01') & (cameo.release_date >= '2022-08-01')]
    tosave = df.loc[tosave.index]
    del tosave['reference']; del tosave['saved']
    tosave = tosave[tosave.seqlen < 750]
    tosave.to_csv('splits/cameo2022.csv')
    
if __name__ == "__main__":
    train_splits()
    apo_splits()
    codnas_splits()
    cameo_splits()