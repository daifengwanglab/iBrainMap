import argparse
import pickle as pk
import scipy.sparse as sp
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def get_common_gene_list(hvg_file, tf_file, gene_list_file):
    """ Function to combine hvg_file & tf_file and identify common genes
        with all genes for this cohort """

    hvgs = pd.read_csv(hvg_file)
    hvgs = list(hvgs.featurekey.values)

    tfs = pd.read_csv(tf_file, header=None)
    tfs = list(tfs[0])

    all_genes = list(set(tfs + hvgs))
    all_genes.sort()

    gene_list = pd.read_csv(gene_list_file, header=None)
    gene_list = list(gene_list[0])

    common_genes = list(set(all_genes).intersection(gene_list))
    common_genes.sort()
    hvgs_new = list(set(hvgs).intersection(common_genes))
    hvgs_new.sort()

    return gene_list, common_genes, hvgs_new

def get_gene_node_features(indiv_df, gene_list=None):
    """ Function to get gene node features - Coexpression values """
    if gene_list is None:
        print('entered')
        gene_list = list(indiv_df.index)

    indiv_df_filt = indiv_df.loc[gene_list, :]
    coexpr = abs(np.corrcoef(indiv_df_filt.T.values, rowvar=False))
    coexpr = pd.DataFrame(coexpr, index=gene_list, columns=gene_list, dtype=np.float16)
    coexpr = coexpr.fillna(0)
    return coexpr

def get_celltype_node_features(indiv_df, meta_data, colNm='subclass', gene_list=None):
    """ Function to get celltype node features - Average gene expression for eaach celltype """
    if gene_list is None:
        gene_list = list(indiv_df.index)

    indiv_df_filt = indiv_df.loc[gene_list, :]
    ctypes = list(set(meta_data[colNm]))
    feat_mat = np.zeros([indiv_df_filt.shape[0], len(ctypes)])

    for idx, ctype in enumerate(ctypes):
        barcodes = list(meta_data.loc[meta_data[colNm]==ctype, 'barcodekey'])
        dat = indiv_df_filt.loc[:, barcodes]
        feat_mat[:, idx] = dat.mean(axis=1)
    scaler = MinMaxScaler()
    feat_mat = scaler.fit_transform(feat_mat)
    feat_mat = pd.DataFrame(feat_mat, index=gene_list, columns=ctypes, dtype=np.float16)
    return feat_mat.T

def get_node_features(args):
    """ Function to extract node features """

    # Get all genes
    all_genes, comm_genes, hv_genes = get_common_gene_list(args.gene_select_file,
                                                           args.tf_list_file, args.gene_name_file)

    # Read gene expression data
    donor_gex = pd.DataFrame(sp.load_npz(args.gene_expr_file).todense()).T
    donor_meta = pd.read_csv(args.cell_meta_data)
    donor_gex.index = all_genes
    donor_gex.columns = donor_meta[args.cell_name_col]

    # Get cell type node features
    ctype_node_feat = get_celltype_node_features(donor_gex, donor_meta,
                                                 colNm=args.cell_group_col, gene_list=hv_genes)
    gene_node_feat = get_gene_node_features(donor_gex, gene_list=comm_genes)
    gene_node_feat = gene_node_feat.loc[:, hv_genes]

    node_features = pd.concat([ctype_node_feat, gene_node_feat])

    with open(args.save+'_NodeFeatures.pkl', 'wb') as fl:
        pk.dump(node_features, fl)

def main():
    """ Main method """
    parser = argparse.ArgumentParser()

    # Input
    parser.add_argument('--gene_expr_file', type=str, default='./snrna_seq_data/M5.npz',
                        help='Path to the gene expression file. Should be a .npz file')
    parser.add_argument('--gene_name_file', type=str, default='./snrna_seq_data/gene_list_MSSM.txt',
                        help='Path to the gnee name files. Should be a .txt file')
    parser.add_argument('--gene_select_file', type=str, help='File for list of genes to be used',
                        default='./snrna_seq_data/NPS_AD_GRN_freeze2.5_FULL.hvg.csv')
    parser.add_argument('--tf_list_file', type=str, help='File containing list of TFs to be used',
                        default='./snrna_seq_data/allTFs_hg38.txt')
    parser.add_argument('--cell_meta_data', type=str, help='File containing cell information',
                        default='./snrna_seq_data/M5_meta_obs.csv')

    parser.add_argument('--cell_name_col', type=str, help='Column name containing cell ids',
                        default='barcodekey')
    parser.add_argument('--cell_group_col', type=str, help='Column name containing cell grouping',
                        default='subclass')

    parser.add_argument('--save', type=str, default='M5',
                        help='File path for the ouput to be saved')
    args = parser.parse_args()
    print(args)

    # Use the run_GRN_SCENIC function to generate the SCENIC GRN per sample
    get_node_features(args)

if __name__ == '__main__':
    main()
