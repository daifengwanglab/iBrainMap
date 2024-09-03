import os
import warnings
import argparse
from sys import exit
import pyscenic
from pyscenic.rss import regulon_specificity_scores
import loompy as lp
import pandas as pd
from arboreto.algo import grnboost2, genie3
from arboreto.utils import load_tf_names
import numpy as np
import scipy.sparse as sc
warnings.filterwarnings('ignore')
from math import sqrt
from scipy.spatial.distance import jensenshannon

def run_GRN_SCENIC(args):
    """ Function to generate RSS and regulon files from scenic"""
    ##Generate GRNboost GRN
    gexpr = sc.load_npz(args.gene_expr_file)
    gexpr = pd.DataFrame(gexpr.toarray())

    # Read the cell and gene names corresponding to the gene expression file
    gene_ids = pd.read_csv(args.gene_name_file, header=None)
    gene_ids = list(gene_ids.iloc[:, 0])

    cellinfo = pd.read_csv(args.cell_meta_data)
    cell_ids = list(cellinfo[args.cell_name_col])

    # Check if the dimension matches
    if (len(cell_ids) != gexpr.shape[0]) | (len(gene_ids) != gexpr.shape[1]):
        print("Gexpr and name length mismatch")
        exit(0)
    gexpr.columns = gene_ids
    gexpr.index = cell_ids

    # Filter genes based on gene list
    gene_select = pd.read_csv(args.gene_select_file, header=None)
    gene_select = list(gene_select.iloc[:, 0])
    common_genes = list(set(gexpr.columns).intersection(gene_select))
    gexpr = gexpr[common_genes[0:100]]

    # Read the Transcription Factor file
    tf_names = pd.read_csv(args.tf_list_file, header=None)
    tf_names = list(tf_names.iloc[:, 0])

    # Run GRN Boost
    #print("Running GRNBoost")
    GRNboost_GRN_fname = args.save + '_grnboost_GRN.csv'
    network = grnboost2(expression_data=gexpr, tf_names=tf_names)
    network.to_csv(GRNboost_GRN_fname,index=False)
    #print("Completed GRNBoost")

    # Generate SCENIC cell type GRN
    f_loom_path_scenic = args.save + "_scenic.loom"

    row_attrs = {
        "Gene": np.array(gexpr.columns) }
    col_attrs = {
        "CellID": np.array(gexpr.index) ,
        "nGene": np.array( np.sum(gexpr.transpose()>0 , axis=0)).flatten() ,
        "nUMI": np.array( np.sum(gexpr.transpose() , axis=0)).flatten() }

    lp.create( f_loom_path_scenic, np.array(gexpr.transpose()), row_attrs, col_attrs)
    reg_path = args.save + '_reg.csv'

    scenic_script = "pyscenic ctx " + GRNboost_GRN_fname + " " + args.feather_path
    scenic_script += " --annotations_fname " + args.motif_path
    scenic_script += " --expression_mtx_fname " + f_loom_path_scenic
    scenic_script += " --output " + reg_path + " --mask_dropouts --num_workers " + str(args.n_cores)
    #print(scenic_script)
    os.system(scenic_script)
    #print('completed scenic_script')

    f_pyscenic_output =  args.save + '_output.loom'

    ps_auc_scrpit = "pyscenic aucell " + f_loom_path_scenic+" " + reg_path
    ps_auc_scrpit += " --output " + f_pyscenic_output + " --num_workers "+ str(args.n_cores)
    #print(ps_auc_scrpit)
    os.system(ps_auc_scrpit)
    #print('completed ps_auc_scrpit')

    ## Generate AUC matrix
    lf = lp.connect(f_pyscenic_output, mode='r+', validate=False)
    auc_mtx = pd.DataFrame( lf.ca.RegulonsAUC, index=lf.ca.CellID)

    auc_file_name = args.save + '_auc_mtx.csv'
    auc_mtx.to_csv(auc_file_name)
    lf.close()

    ##
    rss_cellType = regulon_specificity_scores(auc_mtx, cellinfo[args.cell_group_col])
    rss_cellType.to_csv(args.save + '_celltype_rss.csv')

    # create a dictionary of regulons:
    lf = lp.connect(f_pyscenic_output, mode='r+', validate=False)
    regulons = {}
    for i,r in pd.DataFrame(lf.ra.Regulons,index=lf.ra.Gene).iteritems():
        regulons[i] = list(r[r==1].index.values)
    lf.close()

    #generate regulon_grn
    regulon_list = pd.DataFrame(index = range(2),
                                columns = ['TF', 'gene', 'CoexWeight', 'regulon'])

    for item in regulons:
        if '(+)' in item:
            tf = item.replace('(+)', '')
        else:
            tf = item
        grn_cut = network[network['TF'] == tf]
        grn_cut.index = grn_cut['target']

        if tf in regulons[item]:
            genes = regulons[item]
            genes.remove(tf)
        else:
            genes = regulons[item]

        grn_cut_1 = grn_cut.loc[genes]
        grn_cut_1 = grn_cut_1.rename(columns={'TF': "TF",'target': 'gene',
                                              'importance' : "CoexWeight"})
        grn_cut_1['regulon'] = item
        regulon_list = pd.concat([regulon_list, grn_cut_1])

    regulon_list = regulon_list.tail(-1)
    regulon_list = regulon_list.tail(-1)
    regulon_list.to_csv(args.save + '_regulon_list.csv')

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
    
    # Needed files for running SCENIC 
    parser.add_argument('--motif_path', type=str, default="motifs-v9-nr.hgnc-m0.001-o0.0.tbl",
                        help='File path containing motif information')
    parser.add_argument('--feather_path', type=str, help='Feather File path',
                        default='hg38__refseq-r80__500bp_up_and_100bp_down_tss.mc9nr.genes_vs_motifs.rankings.feather')
    
    parser.add_argument('--n_cores', type=int, help='Number of cores to be used', default=10)
    parser.add_argument('--save', type=str, help='File path for the ouputs to be saved', default='M5')
    args = parser.parse_args()
    print(args)

    # Use the run_GRN_SCENIC function to generate the SCENIC GRN per sample
    run_GRN_SCENIC(args)

if __name__ == '__main__':
    main()
