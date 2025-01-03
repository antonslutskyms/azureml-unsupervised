import argparse
import os
import pandas as pd
import glob
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.cluster import MiniBatchKMeans  # Use MiniBatchKMeans for large datasets
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import SanitizeMol
from rdkit.Chem import MolToSmiles
import json


def init():
    print("Init method called")
    



def run(input_files):
    print("Starting data prep: ", input_files)
    
    # # Find the CSV file in the specified directory
    csv_file = input_files[0]

    print("Csv Files: ", csv_file)

    if csv_file.endswith(".csv"):
        curated_dataset = csv_file
        print(f"Using CSV file: {curated_dataset}")

        df = pd.read_csv(curated_dataset)
        print(f"Original DataFrame shape: {df.shape}")

        def smiles_to_scaffold(smiles):
            mol = Chem.MolFromSmiles(smiles, sanitize=False)  # Avoid stereochemistry issues
            if mol is not None:
                try:
                    # Explicit sanitization before proceeding
                    SanitizeMol(mol)
                    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                    return scaffold
                except Exception as e:
                    print(f"Sanitization failed for SMILES {smiles}: {e}")
            return None
            
        # Generate scaffolds for all molecules in your dataset (using a generator to save memory)
        def generate_scaffolds(smiles_list):
            for smiles in smiles_list:
                scaffold = smiles_to_scaffold(smiles)
                if scaffold is not None:
                    yield Chem.MolToSmiles(scaffold)

        # Convert scaffolds to fingerprints with reduced memory footprint
        def scaffold_to_fingerprint(scaffold, radius=2, n_bits=1024):  # Reduced fingerprint size (n_bits=1024)
            np_array = np.array(AllChem.GetMorganFingerprintAsBitVect(scaffold, radius, nBits=n_bits))
            
            result = np.where(np_array == 1)
            result = result[0]
            return json.dumps({"sz" : len(np_array), "ones" : result.tolist()})
            
        # Efficiently process scaffolds and fingerprints with generators
        def process_fingerprints(scaffold_smiles):
            for smiles in scaffold_smiles:
                scaffold = Chem.MolFromSmiles(smiles)
                if scaffold is not None:
                    fp = scaffold_to_fingerprint(scaffold)
                    yield fp

        print("Generating scaffolds:")
        df.info()

        # Process the data in chunks (for memory efficiency)
        df_col = df.iloc[:, 1]

        scaffold_smiles = list(generate_scaffolds(df_col))  # Process scaffolds using a generator


        # Generate fingerprints and perform KMeans clustering
        fingerprints = list(process_fingerprints(scaffold_smiles))  # Generate fingerprints efficiently


        return fingerprints
    else:
        print("Ignoring file: ", csv_file)
        return []
