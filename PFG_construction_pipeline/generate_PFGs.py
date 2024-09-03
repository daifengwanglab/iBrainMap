import pickle
import argparse
import sys
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
import torch_geometric.transforms as T
import torch
from torch_geometric.data import Data

def run_bio_diffusion(edge_list, gene_list, degree, beta):
    """ Function to generate bio-diffured matrix"""
    tuple_list = [tuple(x) for x in edge_list[['from', 'to']].to_numpy()]

    # Create a graph object from the edgelist
    G = nx.DiGraph(tuple_list)

    # Compute the adjacency matrix of the graph
    adj_matrix = pd.DataFrame(nx.adjacency_matrix(G).toarray())
    node_ids = list(G.nodes())

    adj_matrix.index = node_ids
    adj_matrix.columns = node_ids

    ### Create a diffusion matrix
    diff_mat = pd.DataFrame(np.zeros((len(node_ids), len(node_ids))))
    diff_mat.index = node_ids
    diff_mat.columns = node_ids

    if degree == 1:
        for _, gene in enumerate(gene_list):
            if gene in adj_matrix.index:
                diff_mat[gene][gene]= 1
    elif degree == 2:
        for _, gene in enumerate(gene_list):
            if gene in adj_matrix.index:
                diff_mat[gene][gene]= 1
                n_gene = [n for n in G.neighbors(gene)]
                for n_ge in n_gene:
                    diff_mat[n_ge][n_ge]= 0.7
    elif degree == 3:
        for _, gene in enumerate(gene_list):
            if gene in adj_matrix.index:
                diff_mat[gene][gene] = 1
                n_gene = [n for n in G.neighbors(gene)]
                for n_ge in n_gene:
                    if diff_mat[n_ge][n_ge] == 0:
                        diff_mat[n_ge][n_ge]= 0.7
                    n_gene1 = [n for n in G.neighbors(n_ge)]
                    for n_ge1 in n_gene1:
                        if diff_mat[n_ge1][n_ge1] == 0:
                            diff_mat[n_ge1][n_ge1]= 0.3
    degree_mat = np.diag(np.sum(adj_matrix, axis = 1))
    I = np.eye(len(node_ids))

    # Compute the inferred gene expression levels using the formula
    F_M = beta * np.linalg.inv(I - (1 - beta) * np.dot(adj_matrix, np.linalg.inv(degree_mat)))
    #F_M = 0.5*(F_M + F_M.T)
    F_M = pd.DataFrame(F_M, index=node_ids, columns=node_ids)
    F_dis = F_M@diff_mat

    return F_dis

def add_edge_attr_head(edge_list, F_mat, head_name):
    """ Function to generate edge attribute head """
    # Create an empty graph
    G1 = nx.from_pandas_adjacency(F_mat)
    melt = pd.DataFrame(nx.to_pandas_edgelist(G1))
    melt.columns = ['to', 'from', 'weight']
    #melt = melt['from', 'to', 'weight']
    melt['id'] = melt['from'] + '_' + melt['to']
    melt = melt.set_index('id')

    edge_list1 = edge_list.copy()
    edge_list1['id'] = edge_list1['from'] + '_' + edge_list1['to']
    edge_list1 = edge_list1.set_index('id')

    comm_edge = set(melt.index).intersection(edge_list1.index)
    edge_list1[head_name] = 0
    edge_list1.loc[comm_edge, head_name] = melt.loc[comm_edge]["weight"]

    index_replace = edge_list1[head_name].values <= 1e-5
    edge_list1.loc[index_replace, head_name] = 1e-5

    return edge_list1

def get_default_edge_attr(edge_list, ad_gene_file, scz_gene_file, diffusion_degree=2):
    """Function to diffuse graphs based on priori knowledge and add as edge attributes"""
    ad_gene_list = set(pd.read_csv(ad_gene_file)['x'].values)
    scz_gene_list = set(pd.read_csv(scz_gene_file)['x'].values)

    beta_values = np.arange(0.1, 1, 0.2)
    for idx, beta in enumerate(beta_values):
        diff_ad = run_bio_diffusion(edge_list, ad_gene_list, diffusion_degree, beta)
        if idx ==0:
            edge_list_add = add_edge_attr_head(edge_list, diff_ad,
                                               'D_AD_beta_'+"{:.1f}".format(beta).replace('.', '_'))
        else:
            edge_list_add = add_edge_attr_head(edge_list_add, diff_ad,
                                               'D_AD_beta_'+"{:.1f}".format(beta).replace('.', '_'))

        diff_scz = run_bio_diffusion(edge_list, scz_gene_list, diffusion_degree, beta)
        edge_list_add = add_edge_attr_head(edge_list_add, diff_scz,
                                           'D_SCZ_beta_'+"{:.1f}".format(beta).replace('.', '_'))
    edge_list_add['D_1'] = 1

    return edge_list_add

def get_custom_edge_attr(edge_list, prior_gene_file, diffusion_degree=2):
    """Function to diffuse graphs based on priori knowledge and add as edge attributes"""
    prior_gene_list = pd.read_csv(prior_gene_file)
    prior_gene_list = set(prior_gene_list[0].values)

    beta_values = np.arange(0.1, 1, 0.2)
    for idx, beta in enumerate(beta_values):
        diff_goi = run_bio_diffusion(edge_list, prior_gene_list, diffusion_degree, beta)
        if idx ==0:
            edge_list_add = add_edge_attr_head(edge_list, diff_goi,
                                               'D_GOI_beta_'+"{:.1f}".format(beta).replace('.', '_'))
        else:
            edge_list_add = add_edge_attr_head(edge_list_add, diff_goi,
                                               'D_GOI_beta_'+"{:.1f}".format(beta).replace('.', '_'))
    edge_list_add['D_1'] = 1

    return edge_list_add

def get_edges(cci_file, rss_file, regulon_file, pick_num_tf,
              pick_num_tg, cci_cutoff, rem_celltypes=["EN_NF"]):
    """ Function to merge differnt edges: CT=-CT, CT-TF, CT-TG, TF-TG """

    ccis = pd.read_csv(cci_file)
    ccis.index = ccis.columns
    rss = pd.read_csv(rss_file)
    rss = rss.set_index(rss.columns[0])
    regulons = pd.read_csv(regulon_file)
    regulons = regulons.set_index(regulons.columns[0])

    if ccis.shape[0] == 0:
        return None
    else:
        if rem_celltypes is not None:
            for ctype in rem_celltypes:
                if ctype in rss.index:
                    rss = rss.drop(index=ctype)
                if ctype in ccis.index:
                    ccis = ccis.drop(index=ctype, columns=ctype)

        ### Filter CCIs to get CCI edges based on threshold
        # Scale CCIs to be between 0-1 range
        scaler = MinMaxScaler()
        scaler.fit(ccis.T)
        scaled = scaler.fit_transform(ccis.T)
        scaled_ccis = pd.DataFrame(scaled.T, columns=ccis.columns, index=ccis.columns)
        # Convert matrix into DF with three columns: from, to, and weight
        melted = pd.melt(scaled_ccis.reset_index(), id_vars='index',
                         var_name='column', value_name='edge_weight')
        melted.columns = ['from', 'to', 'org_edge_weight']
        # Filter based on cutoff
        ct_ct = melted[melted['org_edge_weight']> cci_cutoff]
        ct_ct = ct_ct[['from', 'to', 'org_edge_weight']]

        ct_ct['edge_type'] = 'celltype_celltype'

        edge_list = pd.DataFrame()
        for j in range(rss.shape[0]):
            curr_celltype = rss.index[j]

            temp = pd.DataFrame(rss.iloc[j, :])
            temp['Rank'] = temp.iloc[:,0].rank(ascending=False)
            temp = temp.sort_values('Rank')

            ### Construct TF-celltype hub nodes
            # Pick the top TFs based on top x%
            num_tfs = int(np.round(temp.shape[0]*pick_num_tf/100))
            ct_tf = temp.iloc[0:num_tfs, :].reset_index().drop(columns=['Rank'])
            ct_tf['celltype'] = rss.index[j]
            ct_tf.columns = ['TF', 'org_edge_weight', 'celltype']
            ct_tf = ct_tf.loc[:, ['celltype', 'TF', 'org_edge_weight']]
            try:
                ct_tf['TF'] = ct_tf['TF'].str.replace('(+)', '')
            except:
                ct_tf['TF'] = ct_tf['TF'].str.replace('\(\+\)', '')
            ct_tf.columns = ['from', 'to', 'org_edge_weight']
            ct_tf['edge_type'] = 'celltype_TF'

            ### Construct the TF-TG part
            temp_nw = regulons[regulons['TF'].isin(np.unique(ct_tf['to']))]
            temp_nw = temp_nw.sort_values(by='CoexWeight', ascending=False)
            tf_tg = pd.DataFrame()
            for idx, tf_id in enumerate(np.unique(temp_nw['TF'])):
                tmp = temp_nw.loc[temp_nw['TF']==tf_id, :]
                num_tgs = int(np.ceil(tmp.shape[0]*pick_num_tg/100))
                if idx == 0:
                    tf_tg = tmp.iloc[0:num_tgs, :]
                else:
                    tf_tg = pd.concat([tf_tg, tmp.iloc[0:num_tgs, :]])
            tf_tg = tf_tg.loc[:, ['TF', 'gene', 'CoexWeight']]
            tf_tg.columns = ['from', 'to', 'org_edge_weight']
            tf_tg['edge_type'] = 'TF_TG'

            ### Construct the CT-TG part
            ct_tg = pd.DataFrame({'celltype':curr_celltype, 'gene':tf_tg['to']})
            ct_tg.columns = ['from', 'to']
            ct_tg['org_edge_weight'] = 1
            ct_tg['edge_type'] = 'celltype_TG'

            ### Reverse edges
            tf_ct = ct_tf.loc[:, ['to', 'from', 'org_edge_weight', 'edge_type']]
            tf_ct['edge_type'] = 'rev_celltype_TF'
            tf_ct.columns = ['from', 'to', 'org_edge_weight', 'edge_type']

            tg_ct = ct_tg.loc[:, ['to', 'from', 'org_edge_weight', 'edge_type']]
            tg_ct['edge_type'] = 'rev_celltype_TG'
            tg_ct.columns = ['from', 'to', 'org_edge_weight', 'edge_type']

            combined_edges = pd.concat([ct_tf, tf_ct, ct_tg, tg_ct, tf_tg])
            combined_edges['ct_GRN'] = curr_celltype + '_GRN'
            if j ==0:
                edge_list = combined_edges
            else:
                edge_list = pd.concat([edge_list, combined_edges])

        ct_ct['ct_GRN'] = 'cellchat'
        edge_list = pd.concat([edge_list, ct_ct])
        edge_list = edge_list.drop_duplicates()
        return edge_list

def get_node_features(edge_list, node_feat_file, gene_filter_file=None):
    """ Function to extract node features """
    node_feat = pd.read_pickle("M5_NodeFeatures.pkl")
    node_feat = pd.read_pickle(node_feat_file)

    if gene_filter_file is None:
        node_features = node_feat
    else:
        gene_filter_list = pd.read_csv(gene_filter_file)
        gene_filter_list = list(set(gene_filter_list.iloc[0, :].values))

        keep_feat = list(set(node_feat.columns) - set(gene_filter_list))
        node_features = pd.DataFrame(node_feat, columns=keep_feat)

    genes_edgelist = set(edge_list['from'].values) & set(edge_list['to'].values)
    keep_genes = list(set(node_features.index) & genes_edgelist)
    node_features = node_features.loc[keep_genes, :]

    node_ids =  node_features.index
    node_idx = {node:i for i, node in enumerate(node_ids)}

    edge_idx = torch.tensor([[node_idx[edge_list.iloc[i][0]],
                              node_idx[edge_list.iloc[i][1]]]
                             for i in range(edge_list.shape[0])]).t()

    return node_idx, edge_idx, node_features

def generate_pfgs(args):
    """ Function to generate PFGs """
    try:
        # Get combiend edges from CCIs & GRNs
        edge_list = get_edges(args.cci_file, args.rss_file, args.regulon_file,
                              args.num_tf_percent, args.num_tg_percent, 0.5, ['EN_NF'])
        print('edge list generation complete')

        # Get edge attributes using diffusion
        if args.setting =='default':
            edge_attr = get_default_edge_attr(edge_list, args.ad_prior_file,
                                              args.scz_prior_file, args.degree)
        else:
            edge_attr = get_custom_edge_attr(edge_list, args.prior_gene_file, args.degree)
        print('edge attribute generation complete')

        # Get node features
        node_idx, edge_idx, node_feat = get_node_features(edge_list, args.node_feature_file,
                                                          args.gene_filter_file)
        print('node feature generation complete')

        # Remove idolated nodes
        transform = T.Compose([T.remove_isolated_nodes.RemoveIsolatedNodes()])

        data = Data(x=torch.tensor(node_feat.values), edge_index= edge_idx,
                    edge_attr=torch.tensor(edge_attr.iloc[:, 5:].values), pos=None)
        data = transform(data)

        pfg = {}
        pfg['sample_id'] = args.sample_id
        pfg['nodeid'] = node_idx
        pfg['graph'] = data
        pfg['edgelist'] = edge_attr

        with open(args.save + '/' + args.sample_id + '_graph.pkl', "wb") as f:
            pickle.dump(pfg, f)
            print('PFG generation complete')
    except Exception as err:
        print('Could not construct PFGs for this sample')
        print(err)
        sys.exit(0)

def main():
    """ Main method """
    parser = argparse.ArgumentParser()

    # Input
    parser.add_argument('--cci_file', type=str, default='M5_ctype_comm_nw_weight.csv',
                        help='Path to the CCI file generated from step 1')
    parser.add_argument('--rss_file', type=str, default='M5_rss.csv',
                        help='Path to the RSS file generated from step 2')
    parser.add_argument('--regulon_file', type=str, default='M5_regulon.csv',
                        help='Path to regulon file generated from step 2')
    parser.add_argument('--node_feature_file', type=str, default='M5_NodeFeatures.pkl',
                        help='Path to the node feature file from step 3. Must be .pkl file')

    # Gene list files
    parser.add_argument('--gene_filter_file', type=str, default='gene_filter_list.txt',
                        help='File Path to gene list to be removed')
    parser.add_argument('--ad_prior_file', type=str, default='AD_all.csv',
                        help='File path to known AD genes')
    parser.add_argument('--scz_prior_file', type=str, default='SCZ_all.csv',
                        help='File path to known SCZ genes')

    # Additional settings
    parser.add_argument('--num_tf_percent', type=int, default=15,
                        help='Top percent of TFs to be picked for each cell type')
    parser.add_argument('--num_tg_percent', type=int, default=15,
                        help='Top percent of TGs per TF')
    parser.add_argument('--degree', type=int, default=2,
                        help='Diffusion neighborhood degree. Chose between 1, 2, and 3')
    parser.add_argument('--save', type=str, default='.',
                        help='Save file')
    parser.add_argument('--sample_id', type=str, default='M5', help='Save file')

    # Custom setting for graph generation
    parser.add_argument('--setting', type=str, default='default',
                        help='Uses the default settings. Choose between default and custom')
    parser.add_argument('--prior_gene_file', type=str, default='',
                        help='Genes of interest list to be used as biological prior')

    args = parser.parse_args()
    generate_pfgs(args)

if __name__ == '__main__':
    main()
