"""Build the ChEMBL reference fingerprint pickle for novelty checking.

Reads the ChEMBL chemreps TSV (canonical SMILES + InChIKeys), computes
Morgan fingerprints (radius=2, nBits=2048), and writes the reference
pickle via ChEMBLNoveltyChecker.build_reference().

Usage:
    python scripts/build_chembl_reference.py [--input data/chembl_35_chemreps.txt] [--output data/chembl_reference.pkl]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from workbench.molecules.novelty_checker import ChEMBLNoveltyChecker


def generate_tuples(input_path: Path):
    with open(input_path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        count = 0
        failed = 0
        for row in reader:
            smiles = row["canonical_smiles"]
            inchikey = row["standard_inchi_key"]
            if not smiles or not inchikey:
                failed += 1
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                failed += 1
                continue
            fp = GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            count += 1
            if count % 250_000 == 0:
                print(f"  processed {count:,} compounds...", flush=True)
            yield (smiles, inchikey, fp)
        print(f"Done: {count:,} valid, {failed:,} failed parses")


def main():
    parser = argparse.ArgumentParser(description="Build ChEMBL reference pickle")
    parser.add_argument(
        "--input",
        default="data/chembl_35_chemreps.txt",
        help="Path to ChEMBL chemreps TSV file",
    )
    parser.add_argument(
        "--output",
        default="data/chembl_reference.pkl",
        help="Output pickle path",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: {input_path} not found.")
        print("Download it first:")
        print(
            "  wget -O data/chembl_35_chemreps.txt.gz "
            '"https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/'
            'chembl_35/chembl_35_chemreps.txt.gz"'
        )
        print("  gunzip data/chembl_35_chemreps.txt.gz")
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Building reference from {input_path}...")
    ChEMBLNoveltyChecker.build_reference(generate_tuples(input_path), output_path)
    size_mb = output_path.stat().st_size / 1e6
    print(f"Saved to {output_path} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
