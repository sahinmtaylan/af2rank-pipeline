# Barnase–barstar example

`model.pdb` contains protein atoms from chains A (barnase) and D (barstar) of [PDB 1BRS](https://doi.org/10.2210/pdb1BRS/pdb), a two-chain complex. Waters and other chains were removed; for the two alternate-location atoms, conformer A was retained. `target.fasta` contains the corresponding full SEQRES sequences (110 and 89 residues). Four residues lack coordinates in this deposited model, which the cleaning manifest records.

The source PDB archive data are [CC0](https://www.rcsb.org/pages/usage-policy). This is a small, real input for an end-to-end installation test, not a benchmark or a demonstration that a high score establishes binding accuracy.
