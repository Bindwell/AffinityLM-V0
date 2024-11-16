import os
import sys
from Bio import SeqIO   
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

res = []
dir = "/Users/miaanand/Downloads/PP/"


def extract_pdb_kd(text):
    """Extract PDB IDs and convert Kd/Ki values to pKd."""
    # Pattern to match PDB ID and Kd/Ki values
    pattern = r'(\d\w{3})\s+[\d.]+\s+\d{4}\s+(?:Kd|Ki)=(\d+\.?\d*)\s*(pM|nM|uM|mM|fM)'
    matches = re.finditer(pattern, text)
    data = []
    for match in matches:
        pdb_id = match.group(1)
        value = float(match.group(2))
        unit = match.group(3)
        # Convert all values to Molar for pKd calculation
        conversion = {
            'fM': 1e-15,
            'pM': 1e-12,
            'nM': 1e-9,
            'uM': 1e-6,
            'mM': 1e-3
        }
        
        molar = value * conversion[unit]
        pkd = -np.log10(molar)
        
        # Get FASTA sequences for this PDB
        pdb_path = os.path.join(dir, f"{pdb_id.lower()}.ent.pdb")
        protein1_seq = ""
        protein2_seq = ""
        
        if os.path.exists(pdb_path):
            with open(pdb_path, 'r') as pdb_file:
                seq_count = 0
                for record in SeqIO.parse(pdb_file, 'pdb-atom'):
                    if seq_count == 0:
                        protein1_seq = str(record.seq)
                    elif seq_count == 1:
                        protein2_seq = str(record.seq)
                        break
                    seq_count += 1
                    
        data.append({
            'pdb_id': pdb_id,
            'pkd': pkd,
            'protein1_sequence': protein1_seq,
            'protein2_sequence': protein2_seq
        })
    
    return pd.DataFrame(data)


with open("/Users/miaanand/Downloads/PP/index/INDEX_general_PP.2020.txt", "r+") as myfile:
    text = myfile.read()
    print(extract_pdb_kd(text))