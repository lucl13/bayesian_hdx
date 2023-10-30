import pandas as pd


file1 = 'bayesian_hdx_RBSB_1018_APO.dat'
file2 = 'bayesian_hdx_RBSB_1025_APO.dat'


# Read the CSV files
csv1 = pd.read_csv(file1)
csv2 = pd.read_csv(file2)

# Merge the two CSV files on the common columns: start_res, end_res, and peptide_seq
common_peptides = pd.merge(csv1, csv2, on=['start_res', 'end_res', 'peptide_seq'], how='inner')

# Create separate DataFrames for each CSV with only common peptides
common_csv1 = csv1[csv1[['start_res', 'end_res', 'peptide_seq']].apply(tuple, 1).isin(common_peptides[['start_res', 'end_res', 'peptide_seq']].apply(tuple, 1))]
common_csv2 = csv2[csv2[['start_res', 'end_res', 'peptide_seq']].apply(tuple, 1).isin(common_peptides[['start_res', 'end_res', 'peptide_seq']].apply(tuple, 1))]

# Write the common peptides to new CSV files
common_csv1.to_csv(f"{file1.split('.')[0]}_common.dat", index=False)
common_csv2.to_csv(f"{file2.split('.')[0]}_common.dat", index=False)

print('number of peptides in', file1, ':', len(set(csv1['peptide_seq'])))
print('number of common peptides:', len(set(common_csv1['peptide_seq'])))