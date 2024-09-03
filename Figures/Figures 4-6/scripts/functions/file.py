import os
import pickle

import pandas as pd
from tqdm import tqdm


# Locations
DATA_FOLDER = '../../data/'
META = DATA_FOLDER + 'metadata_dec.csv'
GENOTYPE_META = DATA_FOLDER + 'genotype_metadata_dec.csv'
CONTRAST = DATA_FOLDER + 'contrasts.csv'
DOSAGE = DATA_FOLDER + 'PsychAD_Dosage/genotype_varThresh0.5.csv'
COEX_FOLDER = DATA_FOLDER + 'freeze2/regulon_grn/'

# Freeze 2
# ATT = DATA_FOLDER + 'freeze2/attention/homo_5TF_1Tar_graph_with_edgeW_att.pkl'

# Freeze 2.5
ATT_FOLDER = DATA_FOLDER + 'freeze25/c01_5TF_10tar/'
ATT = ATT_FOLDER + 'output_embed_att/homo_c01_5TF_10Tar_p2_5_graph_with_edgeW_att.pkl'
GE = ATT_FOLDER + 'output_embed_att/homo_c01_5TF_10Tar_p2_5_graph_with_edgeW_graph_embedding.pkl'
SID = ATT_FOLDER + 'train_graph/homo_c01_5TF_10Tar_p2_5_graph_with_edgeW_sample_id.pkl'

# Freeze 3
# ATT_FOLDER = DATA_FOLDER + 'freeze3/c15_att_embed_10_16/'
# ATT = ATT_FOLDER + 'c15x_att_Oct_with_edgetype.pkl'
# GE = ATT_FOLDER + 'c15_graph_embedding_all.pkl'
##  SID = NOT PROVIDED

# Freeze 3 (2023-11-06)
# ATT_FOLDER = DATA_FOLDER + 'freeze3/c15_att_Nov/'
# ATT = ATT_FOLDER + 'c15x_att_10pTF_10pTG.pkl'
# GE = NOT PROVIDED
# SID = NOT PROVIDED

# Freeze 3 (2024-02-11)
ATT_FOLDER = DATA_FOLDER + 'freeze3/recent/'
ATT_CSV = ATT_FOLDER + 'MSSM_attn_all_224.csv'
ATT = ATT_FOLDER + 'MSSM_attn_all_224.pkl'
GE = ATT_FOLDER + 'HBCC_10p10p_224_embed_attn.pkl'
# SID = NOT PROVIDED


### File Functions
def get_attention_columns(scaled=False):
    # Load data
    graphs_pkl = get_graphs_pkl()
    graph = graphs_pkl[list(graphs_pkl.keys())[0]]

    # Remove aggregates
    exclude = ['from', 'to', 'from_gene', 'to_gene', 'from_x', 'from_y']

    # Scale
    if not scaled:
        exclude += [c for c in graph.columns if c.endswith('_scale')]
    else:
        exclude += [c[:-6] for c in graph.columns if c.endswith('_scale')]

    # Remove alternatives
    # exclude += [c for c in graph.columns if not c.startswith('att_')]
    exclude += [c for c in graph.columns if not c[-1].isdigit()]

    return [c for c in graph.columns if c not in exclude]


def load_graph_by_id(graph_id, source='attention', column=None, train_omit=True, average=False, **kwargs):
    "Given a subject id `graph_id`, will return dataframe with column(s) `column` from `source`"
    # From individual graphs
    if source == 'coexpression':
        column = 'CoexWeight' if column is None else column
        # Get graph
        graph = pd.read_csv(f'{COEX_FOLDER}{graph_id}_regulon_list.csv')[['TF', 'gene', column, 'regulon']]
        graph = graph.rename(columns={'gene': 'TG', column: 'coef'})  # TF, TG, coef, regulon

    # From pkl
    elif source == 'attention':
        # columns
        # 'att_mean', 'att_max',
        # 'att_D_AD_0_1', 'att_D_AD_0_3', 'att_D_AD_0_5', 'att_D_AD_0_7',
        # 'att_D_no_prior_0', 'att_D_no_prior_1', 'att_D_no_prior_2', 'att_D_no_prior_3'
        column = get_attention_columns()[0] if column is None else column  # Max for retention of head-specific prioritization

        # Load pkl
        graphs_pkl = get_graphs_pkl()

        # Get graph
        if type(column) == type([]):
            # If list, keep many columns and don't standardize name
            graph = graphs_pkl[graph_id][['from', 'to', 'edge_type'] + column]
            graph = graph.rename(columns={'from': 'TF', 'to': 'TG'})
            if average:
                print(graph)
                graph['coef'] = graph.drop(columns=['TF', 'TG', 'edge_type']).mean(axis=1)
                graph = graph[['TF', 'TG', 'edge_type', 'coef']]
        else:
            graph = graphs_pkl[graph_id][['from', 'to', 'edge_type', column]]
            graph = graph.rename(columns={'from': 'TF', 'to': 'TG', column: 'coef'})  # TF, TG, coef

        # Fix names
        # NOTE: Right now these are reversed (or are they?)
        replace = {'celltype_TF': 'rev_celltype_TF', 'rev_celltype_TF': 'celltype_TF'}
        graph['edge_type'] = [replace[et] if et in replace else et for et in graph['edge_type']]

        # Remove training edges
        if train_omit:
            # Remove reverse edges
            graph = graph.loc[graph['edge_type'].apply(lambda s: not s.startswith('rev_'))]
            # Remove TG-celltype edges
            graph = graph.loc[graph['edge_type'] != 'TG_celltype']
        graph = graph.drop(columns='edge_type')

    # Exception
    else:
        raise Exception(f'Source \'{source}\' not found.')

    return graph


def load_graph_embeddings():
    # with open(SID, 'rb') as f:
    #     graph_sids = pickle.load(f)
    with open(GE, 'rb') as f:
        graph_embeddings = pickle.load(f)

    # library = {sid: ge.detach().flatten().numpy() for sid, ge in zip(graph_sids, graph_embeddings)}
    library = {sid: emb.flatten() for sid, emb in zip(graph_embeddings['samples'], graph_embeddings['graph_embeddings'])}
    return library


graphs_pkl = None
def get_graphs_pkl():
    # Load pkl if not already loaded
    global graphs_pkl
    if not graphs_pkl:
        with open(ATT, 'rb') as f:
            graphs_pkl = pickle.load(f)
    return graphs_pkl


def generate_pkl_from_csv(fname=ATT_CSV):
    "Generate `graphs_pkl` file from source csv file."
    global graphs_pkl

    # Load csv
    full_csv = pd.read_csv(fname)
    full_csv = full_csv.set_index('id')

    # Formulate pkl
    graphs_pkl = {}
    for sid in tqdm(full_csv['sample'].unique()):
        graphs_pkl[sid] = full_csv.loc[full_csv['sample'] == sid].drop(columns='sample')

    # Dump pickle
    with open(ATT, 'wb') as f:
        pickle.dump(graphs_pkl, f, protocol=pickle.HIGHEST_PROTOCOL)


def get_meta():
    return pd.read_csv(META)


def get_genotype_meta():
    return pd.read_csv(GENOTYPE_META)


def get_dosage():
    dosage = pd.read_csv(DOSAGE)
    dosage = dosage.set_index('snp_id')
    dosage = dosage.loc[dosage.index != 'snp_id']
    return dosage


contrast_table = None
def get_contrast(contrast):
    # Load if not already loaded
    global contrast_table
    if contrast_table is None:
        contrast_table = pd.read_csv(CONTRAST, dtype=str)

    # Construct dictionary
    library = {}
    for sub in pd.unique(contrast_table[contrast]):
        if pd.isna(sub): continue
        library[sub] = list(contrast_table['SubID'].loc[contrast_table[contrast]==sub])

    return library


def load_many_graphs(subject_ids, **kwargs):
    "Load as many graphs from `subject_ids` as available"
    graphs = []
    sids = []
    for sid in subject_ids:
        try:
            graphs.append(load_graph_by_id(sid, **kwargs))
            sids.append(sid)
        except: pass

    return graphs, sids


def get_enrichment(fname, num_diseases=10):
    # Skip if doesn't exist
    if not os.path.isfile(fname + '.csv'):
        return

    # Load file
    enrichment = pd.read_csv(fname + '.csv')
    enrichment = (
        enrichment[['Description', '_LogP_MyList']]
        .rename(columns={'Description': 'Disease', '_LogP_MyList': '-log(p)'})
    )
    enrichment['-log(p)'] *= -1
    enrichment['Cell Type'] = 'All'

    # Sort and filter to top 10
    # TODO: Filter to common diseases across cell types
    enrichment = enrichment.sort_values('-log(p)', ascending=False).iloc[:num_diseases]

    # Return
    return enrichment
