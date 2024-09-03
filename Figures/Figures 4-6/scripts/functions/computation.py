import itertools as it
import warnings

import graph_tool.all as gt
import networkx as nx
import numpy as np
import pandas as pd
import pingouin as pg
import scipy.stats
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .file import *
from .utility import *


### Computational Functions
def compute_statistics(meta, x, x_sub, filter=.8, **kwargs):
    # Initialize graph summary list
    graph_summary_list = []

    # Get unique values
    unique_x = meta[x].unique()
    unique_x_sub = meta[x_sub].unique()

    # Calculate per graph
    print('Calculating statistics...')
    for val, val_sub in tqdm(it.product(unique_x, unique_x_sub), total=len(unique_x)*len(unique_x_sub)):
        graph_ids = list(meta[(meta[x]==val)*(meta[x_sub]==val_sub)]['SubID'])

        num_samples = len(graph_ids)  # 10
        for graph_id in np.random.choice(graph_ids, min(len(graph_ids), num_samples), replace=False):
            # Load graph
            try:
                graph = load_graph_by_id(graph_id, **kwargs)
            except:
                continue

            # Filter synthetic cells
            if 'source' in kwargs and kwargs['source'] == 'attention':
                synthetic_vertices = list(graph[graph['TG'] == 'hub']['TF'])
            else:
                synthetic_vertices = []

            # TODO: Filter by coef, cell-type regulons

            # Results
            results_list = [graph_id, val, val_sub]

            # Filter to high coef
            if filter:
                graph = graph[graph['coef'] >= graph['coef'].quantile(filter)]

            # Calculate TF outgoing
            tf_outgoing = np.mean([(graph['TF']==gene).sum() for gene in graph['TF'].unique() if gene not in synthetic_vertices])
            results_list.append(tf_outgoing)

            # Calculate TG outgoing (Long Runtime)
            tg_outgoing = 0  # np.mean([(graph['TG']==gene).sum() for gene in graph['TG'].unique() if gene not in synthetic_vertices])
            results_list.append(tg_outgoing)

            # Create graph
            graph_nx = nx.from_pandas_edgelist(graph, 'TF', 'TG', 'coef')

            # Calculate TF closeness
            tf_closeness = np.mean([nx.closeness_centrality(graph_nx, u=gene) for gene in graph['TF'].unique() if gene not in synthetic_vertices])
            results_list.append(tf_closeness)

            # Get cliques
            cliques = sum([1 for _ in nx.find_cliques(graph_nx)])
            results_list.append(cliques)

            # Record results
            graph_summary_list.append(results_list)

    # Create summary df
    graph_summary = pd.DataFrame(graph_summary_list, columns=[
        'Graph ID',
        x,
        x_sub,
        'TF Outgoing',
        'TG Outgoing',
        'TF Closeness',
        'Cliques',
    ])

    # Return
    return graph_summary


def compute_graph(graph, filter=0):  # TODO: Find something more interesting than cutting off `filter`
    # Filter to high-valued edges
    if filter:
        graph = graph[graph['coef'] >= graph['coef'].quantile(filter)]

    # Convert graph to graph_tools
    list_of_tuples = list(graph.itertuples(index=False, name=None))
    g = gt.Graph(list_of_tuples, hashed=True, eprops=[('coef', 'double')])

    # Label self loops and add color
    g.ep.self_loop = g.new_edge_property('bool')
    gt.label_self_loops(g, eprop=g.ep.self_loop, mark_only=True)
    g.vp.self_loop_value = g.new_vertex_property('double')
    g.ep.color = g.new_edge_property('vector<double>')
    for e in g.edges():
        # Label self-loops
        if g.ep.self_loop[e]:
            g.vp.self_loop_value[e.source()] = g.ep.coef[e]

        # Add color to edges
        alpha = get_alpha(g.ep.coef[e])
        g.ep.color[e] = [0, 0, 0, alpha]

    # Determine color and flavor text
    g = assign_vertex_properties(g)

    # View without self-loops
    g_noself = gt.GraphView(g, efilt=lambda e: not g.ep.self_loop[e])

    return g_noself


def compute_edge_summary(graphs=None, concatenated_graph=None, *, subject_ids, min_common_edges=1, threshold=None):
    "Return edge x subject dataframe for all graphs"
    # TODO: Make `subject_ids` not required
    # Setup
    assert graphs is not None or concatenated_graph is not None
    if graphs is not None: concatenated_graph = concatenate_graphs(*graphs, threshold=threshold)

    # Format edge weights
    # df = pd.DataFrame(columns=['Edge']+subject_ids)
    df = {k: [] for k in ['Edge']+subject_ids}
    print('Collecting edges...')
    for e in tqdm(concatenated_graph.edges(), total=concatenated_graph.num_edges()):
        edge_name = get_edge_string(concatenated_graph, e)
        coefs = concatenated_graph.ep.coefs[e]
        # Take only edges which are common between two or more graphs
        if sum([c!=0 for c in coefs]) >= min_common_edges:
            row = [edge_name] + list(coefs)
            # df.loc[df.shape[0]] = row  # Slow
            for k, v in zip(df, row):
                df[k].append(v)
    df = pd.DataFrame(df)

    # Find variance and mean
    df['Variance'] = np.var(df.iloc[:, 1:1+len(coefs)], axis=1)
    df['Mean'] = np.mean(df.iloc[:, 1:1+len(coefs)], axis=1)
    # df = df.sort_values(['Variance', 'Mean'], ascending=[True, False])
    # df['index'] = list(range(len(df)))  # Store sort

    return df, concatenated_graph


def compute_aggregate_edge_summary(contrast_subject_ids, *, column, max_graphs=np.inf, threshold=None):
    "Return concatenated graphs for a contrast"
    # TODO: Add list input for `contrast_subject_ids`
    # For each subgroup of the contrast
    contrast_concatenated_graphs = {}; contrast_concatenated_subject_ids = {}
    for key, subject_ids in contrast_subject_ids.items():  # , total=len(contrast_subject_ids)
        # Get concatenated graph
        graphs = []; sids = []
        for sid in np.random.choice(subject_ids, len(subject_ids), replace=False):  # , total=len(subject_ids)
            try: graphs.append(compute_graph(load_graph_by_id(sid, column=column)))
            except: continue
            sids.append(sid)
            if len(sids) >= max_graphs: break
        # Skip if no graphs found
        if len(sids) == 0:
            continue
        concatenated_graph = concatenate_graphs(*graphs, threshold=threshold)
        contrast_concatenated_graphs[key] = concatenated_graph
        contrast_concatenated_subject_ids[key] = sids

        # Cleanup
        del graphs

    return contrast_concatenated_graphs, contrast_concatenated_subject_ids


def compute_contrast_summary(contrast, *, column, population=True):
    "Return dataframe with edge mean and variance for all subgroups in contrast"
    # TODO: Allow for multiple columns at the same time, requires overhaul of multiple functions, including concat...
    # Get subgroup variance
    contrast_concatenated_graphs, contrast_concatenated_subject_ids = compute_aggregate_edge_summary(
        contrast_subject_ids=contrast, column=column)  # Threshold included for reliable variance calculation
    df_subgroup = {}
    for subgroup in contrast_concatenated_graphs:
        df, concatenated_graph = compute_edge_summary(
            concatenated_graph=contrast_concatenated_graphs[subgroup], subject_ids=contrast_concatenated_subject_ids[subgroup])
        df = df[['Edge', 'Mean', 'Variance']]
        df['Subgroup'] = subgroup
        df_subgroup[subgroup] = df

    # Get population variance
    if population:
        sample_ids = {'Population': sum([contrast[k] for k in contrast], [])}
        contrast_concatenated_graphs, contrast_concatenated_subject_ids = compute_aggregate_edge_summary(
            contrast_subject_ids=sample_ids, column=column)  # Threshold included here for filtering
        df, concatenated_graph = compute_edge_summary(
            concatenated_graph=contrast_concatenated_graphs['Population'], subject_ids=contrast_concatenated_subject_ids['Population'])
        df = df[['Edge', 'Mean', 'Variance']]
        df['Subgroup'] = 'Population'
        df_subgroup['Population'] = df  # Add population as subgroup

    return df_subgroup


def compute_BRAAK_comparison(
        contrast,
        *,
        meta,
        column,
        target='BRAAK_AD',
        edges_include=None,
        edge_percentile=90,
        num_edges=1,
        seed=42):
    """
    Compute df with attention scores of `num_edges` random edges with edge commonality in the
    `edge_percentile` percentile annotated by `target` in `meta` over all individuals in the
    contrast.
    """
    # Calculate
    sids = sum([sids for _, sids in contrast.items()], [])
    all_graphs, sids = load_many_graphs(sids, column=column)
    all_graphs = [compute_graph(graph) for graph in all_graphs]
    df, _ = compute_edge_summary(graphs=all_graphs, subject_ids=sids)

    # Process
    df = df.drop(columns=['Variance', 'Mean'])
    df = pd.melt(df, id_vars=['Edge'], var_name='Subject ID', value_name='Attention')
    df.index = df['Subject ID']
    df_meta = meta.copy()[[target]]
    df_meta.index = meta['SubID']
    df = df.join(df_meta, how='left').reset_index(drop=True)

    # Format
    df = df.loc[df['Attention'] != 0]  # Remove 0 attention
    if edges_include is None:
        all_possible_edges, counts = np.unique(df['Edge'], return_counts=True)
        all_possible_edges = all_possible_edges[counts > np.percentile(counts, edge_percentile)]
        # TODO: Use highest variance or similar rather than random edges
        np.random.seed(seed)
        edges_include = np.random.choice(all_possible_edges, num_edges, replace=False)
    df = df.loc[[e in edges_include for e in df['Edge']]]

    return df, edges_include


def compute_prediction_confusion(
        contrast,
        *,
        meta,
        column,
        prioritized_edges,
        target='BRAAK_AD',
        classifier_type='SGD',
        random_state=42):
    # Calculate
    sids = sum([sids for _, sids in contrast.items()], [])  # All sids in contrast
    all_graphs, sids = load_many_graphs(sids, column=column)
    all_graphs = [compute_graph(graph) for graph in all_graphs]
    df, concatenated_graph = compute_edge_summary(graphs=all_graphs, subject_ids=sids)

    # Filter
    df = df.drop(columns=['Variance', 'Mean'])
    df = df.loc[[e in prioritized_edges for e in df['Edge']]]

    # Format
    X = np.array(df)[:, 1:].T
    df_meta = meta.copy()
    df_meta.index = df_meta['SubID']
    df_meta = df_meta.loc[list(df.columns)[1:]].reset_index(drop=True)
    y = df_meta[target].to_numpy().astype(str)

    # Remove nan
    is_nan = pd.isna(df_meta[target])
    X, y = X[~is_nan], y[~is_nan]

    # Remove classes with too few samples
    # NOTE: This prevents an error with multiclass prediction for sklearn
    min_samples = 2
    y_unique, y_counts = np.unique(y, return_counts=True)
    indices_to_include = sum([y==val for val, count in zip(y_unique, y_counts) if count >= min_samples])
    indices_to_include = np.array(indices_to_include).astype(bool)
    X, y = X[indices_to_include], y[indices_to_include]

    # Formatting and object standardization
    names = np.unique(y)

    # Predict
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=random_state)
    if classifier_type == 'SGD':
        classifier = SGDClassifier(random_state=random_state).fit(X_train, y_train)
    elif classifier_type == 'MLP':
        classifier = MLPClassifier(random_state=random_state).fit(X_train, y_train)
    else:
        raise ValueError(f'Classifier type {classifier_type} not found.')
    accuracy = classifier.score(X_test, y_test)

    # Evaluate
    confusion_matrix = metrics.confusion_matrix(y_test, classifier.predict(X_test), labels=names)
    row_sum = confusion_matrix.sum(axis=1)
    row_acc = np.diag(confusion_matrix) / row_sum
    col_sum = confusion_matrix.sum(axis=0)
    col_acc = np.diag(confusion_matrix) / col_sum
    df = pd.DataFrame(
        confusion_matrix,
        # Predicted names
        columns=[
            f'Predicted {name} (n={int(n)}, acc={racc:.3f})'
            for name, n, racc in zip(names, col_sum, col_acc)],
        # True names
        index=[
            f'{name} (n={int(n)}, acc={racc:.3f})'
            for name, n, racc in zip(names, row_sum, row_acc)])

    return df, accuracy


def compute_head_comparison(subject_ids, **kwargs):
    "Return edge x head difference DataFrame"
    # Setup
    all_columns = get_attention_columns()

    # Get graphs
    joined_graphs = get_many_graph_lists(subject_ids, all_columns)

    # CLI
    print(f'{joined_graphs.shape[0]} common edges found')

    # Calculate differences
    for column in all_columns:
        joined_graphs[column] = joined_graphs[column+'_s1'] - joined_graphs[column+'_s2']

    # Get top idx
    idx_to_include = get_top_idx(joined_graphs.abs(), all_columns, **kwargs)

    # Filter
    joined_graphs = joined_graphs.iloc[idx_to_include][all_columns]

    return joined_graphs


def get_graphs_from_sids(subject_ids, *, method='attention', column=None):
    # TODO: Add try, except
    if method == 'coex':
        return [
            cull_isolated_leaves(
                compute_graph(
                    scale_edge_coefs_list(
                        load_graph_by_id(
                            sid,
                            source='coexpression'),
                        1./60),
                    filter=.9))
            for sid in subject_ids]
    elif method == 'attention':
        return [compute_graph(load_graph_by_id(sid, column=column)) for sid in subject_ids]


def compute_attention_dosage_correlation(
        dosage,
        *,
        meta,
        subject_ids,
        column=None,
        target_edge=None,
        return_target_edge=None,
        chromosomes=None):
    """
    Computes attention-dosage correlation across many subjects and SNPs

    If `chromosomes` is included, filters visualization to chromosomes in `chromosomes`
    """
    # TODO: Allow for multiple target edges
    # Parameters
    if return_target_edge is None: return_target_edge = target_edge is None

    # Non-destructive
    dosage = dosage.copy()

    # Convert dosage ids to subjects
    print('Converting dosage ids to subject ids...')
    dosage_ids = np.array(dosage.columns)
    meta_snp_columns = [s for s in meta.columns if 'SNP' in s.upper()]
    dosage_subject_ids = []
    snp_type = []
    for dosage_id in dosage_ids:
        # Find idx and column
        for col in meta_snp_columns:
            idx = np.argwhere(meta[col] == dosage_id)
            if len(idx) > 0: break
        # Error on completion
        else:
            warnings.warn(f'Unable to find SNP \'{dosage_id}\' in metadata')
            dosage_subject_ids.append(None)
            snp_type.append(None)
            continue
        assert len(idx) == 1, f'Duplicate SNPs \'{dosage_id}\' found in metadata'
        idx = idx[0][0]

        # Record
        subject_id = meta['SubID'].iloc[idx]
        dosage_subject_ids.append(subject_id)
        snp_type.append(col)
    dosage.columns = dosage_subject_ids

    # Find intersection
    intersection_ids = list(set(subject_ids).intersection(set(dosage.columns)))
    dosage = dosage[intersection_ids]

    # Load graphs based on intersection
    # TODO: Use something like `compute_edge_summary(graphs=get_graphs_from_sids(graph_sids, column=column), subject_ids=graph_sids)[0]`
    #   to get full array
    graphs, graph_sids = load_many_graphs(intersection_ids, column=column)

    # Get edge to use for correlation analysis
    # TODO: Replace this with something meaningful
    if target_edge is None:
        print('Selecting target edge...')
        ## First edge
        # target_edge = get_edge_string(list(graphs[0][0].loc[0, ['TF', 'TG']]))
        ## Most variant edge
        variance = compute_edge_summary(graphs=get_graphs_from_sids(graph_sids, column=column), subject_ids=graph_sids)[0]
        variance = variance.set_index('Edge')
        variance = variance.std(axis=1)
        target_edge = variance.index[np.argmax(variance)]
    target_edge = target_edge.split(get_edge_string(['', '']))

    # Get attention for the edge over all subject ids in `intersection_ids`
    attention, attention_sids = [], []
    for graph, sid in zip(graphs, graph_sids):
        # Find target edge
        mask = graph.apply(lambda x: (x['TF'] == target_edge[0]) * (x['TG'] == target_edge[1]), axis=1)
        idx = np.argwhere(mask)

        # If not found, skip graph
        if len(idx) < 1: continue
        idx = idx[0][0]

        # Record
        value = graph['coef'].iloc[idx]
        attention.append(value)
        attention_sids.append(sid)
    attention = np.array(attention)
    attention_sids = np.array(attention_sids)

    # Compute correlation over all SNPs
    print('Computing correlations...')
    chrs = []
    coords = []
    corrs = []
    for snp_id in tqdm(dosage.index):
        # Get snp information
        chr, coord, _, _ = get_genomic_coordinates(snp_id)

        # This filtering could be done earlier, but this seemed more harmonious
        if chromosomes is not None and chr not in [f'chr{s}' for s in chromosomes]:
            continue

        # Format dosage for correlation analysis
        snp_dosage = dosage.loc[snp_id, attention_sids].to_numpy().astype(float)

        # Compute correlation between attention and dosage
        ## Numpy
        # corr = np.corrcoef(attention, snp_dosage)[0, 1]
        ## Scipy
        corr = scipy.stats.pearsonr(attention, snp_dosage)[1]

        # Record
        chrs.append(chr)
        coords.append(coord + get_chromosome_coordinate(chr[3:]))  # Get absolute coordinate
        corrs.append(corr)

    # Transform p values
    ## FDR adjustment
    corr = scipy.stats.false_discovery_control(corr)
    ## Change scale
    corr = -np.log(corr)  # -log(p)

    # Format
    df = pd.DataFrame({
        'Chromosome': chrs,
        'Genomic Coordinate (bp)': coords,
        '-log(FDR-Adjusted Correlation p-value)': corrs})

    # Return
    ret = (df,)
    if return_target_edge: ret += (get_edge_string(target_edge),)
    return format_return(ret)


def compare_graphs_enrichment(g1, g2, *, sid_1, sid_2, nodes, include_tgs=True, threshold=.5):
    def get_tfs(g, cell_type):
        # Locate cell type vertex
        v_cell = gt.find_vertex(g, g.vp.ids, cell_type)
        if len(v_cell) == 0: return []
        v_cell = v_cell[0]

        # Get TFs
        tfs = [v for v in v_cell.in_neighbors() if not string_is_synthetic(g.vp.ids[v]) and g.ep.coef[g.edge(v, v_cell)] > threshold]
        # Add TGs (Maybe once graph is larger this won't be needed)
        if include_tgs:
            tgs = [w for v in tfs for w in v.out_neighbors() if not string_is_synthetic(g.vp.ids[w]) and g.ep.coef[g.edge(v, w)] > threshold]
            tfs = tfs + tgs

        tfs = np.unique(tfs)

        # Return
        return [g.vp.ids[v] for v in tfs]

    df = pd.DataFrame()
    for cell_type in nodes:
        tfs_1 = get_tfs(g1, cell_type)
        tfs_2 = get_tfs(g2, cell_type)
        unique_tfs_1 = set(tfs_1) - set(tfs_2)
        unique_tfs_2 = set(tfs_2) - set(tfs_1)

        # Save to df
        df_new = pd.DataFrame()
        df_new[f'{sid_1} - {cell_type}'] = list(unique_tfs_1)
        df = df.join(df_new, how='outer')
        df_new = pd.DataFrame()
        df_new[f'{sid_2} - {cell_type}'] = list(unique_tfs_2)
        df = df.join(df_new, how='outer')

        # Save to file
        # with open(f'../plots/disgenet.{cell_type}.{subject_id_1}.txt', 'w') as f:
        #     f.write('\n'.join(list(unique_tfs_1)))
        # with open(f'../plots/disgenet.{cell_type}.{subject_id_2}.txt', 'w') as f:
        #     f.write('\n'.join(list(unique_tfs_2)))

    return df


def format_enrichment(enrichment, term_filter=filter_go_terms, num_filter=None, replace_dot=False, show_go_num=False):
    # Filter using term filter
    if term_filter is not None:
        mask = term_filter(enrichment['GO'].to_list())
        enrichment = enrichment.iloc[mask].copy()

    # Add GO term to description
    if show_go_num: enrichment['Description'] = enrichment.apply(lambda r: f'({r["GO"]}) {r["Description"]}', axis=1)

    # Format
    keep = [c for c in enrichment.columns if c.startswith('_LogP_')] + ['Description']
    def rename(n):
        if n.startswith('_LogP_'):
            return n[len('_LogP_'):]
        return n
    enrichment = enrichment[keep].rename(columns=rename)
    enrichment = enrichment.melt(id_vars='Description', var_name='Gene Set', value_name='-log10(p)')
    # enrichment = enrichment.join(
    #     pd.DataFrame(enrichment['Gene Set'].apply(lambda x: x.split('.')).tolist(), columns=('Subject', 'Cell Type')))
    if replace_dot: enrichment['Gene Set'] = enrichment['Gene Set'].apply(lambda s: s.replace('.', ' '))
    enrichment['-log10(p)'] = -enrichment['-log10(p)']
    enrichment = enrichment.loc[enrichment['-log10(p)'] != 0]
    # df = enrichment.pivot(index='Description', columns='Gene Set', values='-log10(p)').fillna(0)

    # Filter to top `num_filter` descriptions for each gene set by sum across all gene sets
    if num_filter is not None:
        rank_by_sum = (
            enrichment.groupby(('Description'))['-log10(p)']
            .sum().rank(ascending=False)
            [enrichment['Description']])
        for gs in np.unique(enrichment['Gene Set']):
            enrichment.loc[enrichment['Gene Set']==gs] = enrichment.loc[enrichment['Gene Set']==gs].loc[
                list(rank_by_sum.loc[list(enrichment['Gene Set']==gs)].rank() <= num_filter)]
        enrichment = enrichment.dropna()

    return enrichment


def get_module_scores(g):
    "Compute module scores"
    association = []
    name = []
    score = []
    for v in g.vertices():
        # Escape if not TF
        if 'tf' not in g.vp.node_type[v]: continue
        # Get association
        association_list = None
        for e in v.in_edges():
            v_source = e.source()
            # If synthetic, record
            if 'celltype' == g.vp.node_type[v_source]:
                if association_list is None: association_list = [g.vp.ids[v_source]]
                else: association_list += [g.vp.ids[v_source]]

        # Get scores
        for e in v.out_edges():
            v_target = e.target()
            # Escape if not TG
            if 'tg' not in g.vp.node_type[v_target]: continue
            # Record weights
            for assoc in association_list:
                association.append(assoc)
                name.append(g.vp.ids[v])
                score.append(g.ep.coef[e])

    df = pd.DataFrame({
        'Cell Type': association,
        'TF': name,
        'Module Score': score,
    }).assign(TGs=1).groupby(['Cell Type', 'TF']).sum().reset_index()

    # Append #TGs to TF names
    df['TF'] = df.apply(lambda r: f'{r["TF"]} ({r["TGs"]})', axis=1)
    df = df.drop(columns='TGs')

    return df


def compute_edge_counts(
        data,
        *,
        edges,
        heads,
        threshold=90,
        threshold_type_override=None,
        **kwargs,
    ):
    if threshold_type_override == 'raw': pass  # Input threshold directly
    elif threshold_type_override == 'max' or threshold is None:
        # Threshold by max/10 on head
        threshold = np.nanmax(data, axis=(0, 2)).reshape((1, -1, 1)) / 10
    elif threshold_type_override == 'percentile' or threshold >= 1:
        # Percentile threshold (integer)
        # NOTE: Functions like `np.nanpercentile` crash for large arrays
        threshold = []
        for i in range(data.shape[1]):
            vals = data[:, i, :].flatten()
            vals = vals[~np.isnan(vals)]
            threshold.append(np.percentile(vals, 10))
        threshold = np.array(threshold).reshape((1, -1, 1))

    # Apply
    within_range = data > threshold

    # Get counts for edges
    counts = within_range.sum(axis=2)
    counts = pd.DataFrame(counts, index=edges, columns=heads)

    # Melt and format
    counts = counts.reset_index(names='Edge').melt(id_vars='Edge', var_name='Head', value_name='Count')
    # print(counts.loc[counts['Count'] > 0])

    # Return
    return counts


def compute_individual_genes(
    counts,
    *,
    columns=None,
    percentage_prioritizations_range=(.045, .055),
    num_subjects,
    column_names=None,
):
    # Defaults
    if columns is None: columns = get_attention_columns()

    # Remove low counts (was zero, but far too many were low)
    # counts = counts.loc[counts['Count'] > 1]

    # Filter and rename
    counts = counts.loc[counts['Head'].isin(columns)]
    if column_names is not None: counts.loc[:, 'Head'] = counts['Head'].map(lambda s: column_names[np.argwhere(np.array(columns)==s)[0][0]])

    # Determine edges that are highly individual for enrichment (between `percentile_prioritizations_range`%s)
    individual_genes = counts.loc[(counts['Count'] >= (percentage_prioritizations_range[0]*num_subjects)) * (counts['Count'] < (percentage_prioritizations_range[1]*num_subjects))]
    individual_genes = individual_genes.copy()
    # Parse into genes from edges
    individual_genes['Edge'] = individual_genes['Edge'].map(lambda s: split_edge_string(s))
    individual_genes = individual_genes.drop(columns='Edge').reset_index(drop=True).join(pd.DataFrame(individual_genes['Edge'].to_list(), columns=('TF', 'TG')))
    individual_genes = individual_genes.melt(id_vars=['Head', 'Count'], var_name='Gene Type', value_name='Gene').drop(columns=['Gene Type', 'Count']).drop_duplicates()
    # Filter synthetic
    individual_genes = individual_genes.loc[individual_genes['Gene'].apply(lambda s: not string_is_synthetic(s))]

    return individual_genes


def compute_differential_genes_from_sids(sids, *, column, vertex_ids=None, threshold=.1):
    # Formatting
    subject_id_1, subject_id_2 = sids

    # NOTE: Column doesn't matter here with the current snipping method
    g1 = compute_graph(load_graph_by_id(subject_id_1, column=column))
    g2 = compute_graph(load_graph_by_id(subject_id_2, column=column))

    # Get unique modules
    df = compare_graphs_enrichment(g1, g2, sid_1=subject_id_1, sid_2=subject_id_2, nodes=vertex_ids, threshold=threshold)

    return df



def compute_all_important_genes_from_sids(
        sids,
        *,
        data,
        edges,
        heads,
        columns=None,
        vertex_ids=None,
        threshold=.01,
        column_names=None,
        percentage_prioritizations_ranges=[(center-center/10, center+center/10) for center in (.01, .05, .1)],
        **kwargs):
    # Defaults
    if columns is None: columns = get_attention_columns()

    # Get differential genes
    print('Computing differential attention genes...')
    df = pd.DataFrame()
    for i, column in tqdm(enumerate(columns), total=len(columns)):
        df_new = compute_differential_genes_from_sids(sids, column=column, vertex_ids=vertex_ids, threshold=threshold)
        df_new = df_new.rename(columns=lambda s: f'{s} - {column if column_names is None else column_names[i]}')
        df = df.join(df_new, how='outer')

    # Add individually important edges
    print('Computing individually variant genes...')
    counts = compute_edge_counts(data, edges=edges, heads=heads, threshold=threshold)
    for ppr in percentage_prioritizations_ranges:
        individual_genes = compute_individual_genes(counts, num_subjects=data.shape[2], percentage_prioritizations_range=ppr, columns=columns, column_names=column_names)
        for column in np.unique(individual_genes['Head']):
            df_new = pd.DataFrame(
                individual_genes.loc[individual_genes['Head']==column, 'Gene'].to_list(),
                columns=(f'{column} - {100*ppr[0]:.1f}-{100*ppr[1]:.1f}%',))
            df = df.join(df_new, how='outer')

    # Remove empty columns
    df = df.loc[:, (~df.isna()).sum(axis=0) > 0]

    # Add background (formatted for Metascape)
    all_genes = [node for node in np.unique([s.split(get_edge_string()) for s in edges]) if not string_is_synthetic(node)]
    df_new = pd.DataFrame({'_BACKGROUND': all_genes})
    df = df.join(df_new, how='outer')

    return df


def compute_prs_difference(
        meta,
        *,
        data,
        edges,
        heads,
        subject_ids,
        subsample=1.,
        prs_col='prs_scaled_SCZ.3.5_MVP',
        covariates=None,
        random_seed=42,
        **kwargs,
    ):
    "Compute PRS correlation across edges and heads"

    # Format
    if covariates is not None: covariates = covariates.copy().set_index('SubID')  # Don't alter original

    # Get PRS scores
    # NOTE: There are several prs_SCZ columns, which to use is unclear
    prs = meta.set_index('SubID').loc[subject_ids, prs_col]

    # Filter to subjects with PRS scores
    present_id_mask = (~prs.isna()).to_numpy()
    data = data[:, :, present_id_mask]
    subject_ids = np.array(subject_ids)[present_id_mask]
    prs = prs.loc[~prs.isna()].to_numpy()

    # Compute correlations
    df = pd.DataFrame(columns=['edge', 'head', 'n', 'correlation', 'p'])
    if subsample != 1.: np.random.seed(random_seed)
    df_corr = pd.DataFrame(prs, index=subject_ids, columns=['prs'])  # PRS
    df_corr = df_corr.join(covariates)  # Join covariates
    for e, h in tqdm(it.product(range(len(edges)), range(len(heads))), total=len(edges)*len(heads)):
        # Random sample (for testing)
        if subsample != 1.:
            if np.random.rand() > subsample: continue

        # Acquire and filter to present data
        data_eh = data[e, h]

        # Create data mask
        # present_data_mask = ~np.isnan(data_eh)
        # Skip if not enough data
        # if present_data_mask.sum() < 2 or data_eh[present_data_mask].std() == 0 or prs[present_data_mask].std() == 0: continue
        # Compute correlation
        # TODO: Compute using matrix
        # corr, p = scipy.stats.pearsonr(data_eh[present_data_mask], prs[present_data_mask])  # Sensitive to outliers
        # corr, p = scipy.stats.spearmanr(data_eh[present_data_mask], prs[present_data_mask])

        # Add/replace attentions
        df_corr['attention'] = data_eh
        # Skip if not enough data
        n_data = (~df_corr.isna()).min(axis=1).sum()
        if (~df_corr.isna()).min(axis=1).sum() < 20: continue  # Minimum for calculating p and CI is 14 data points, but will still throw (sampling?) errors
        # Compute partial correlation
        stats = pg.partial_corr(data=df_corr.reset_index(drop=True), x='attention', y='prs', covar=covariates.columns.to_list(), method='spearman')
        corr, p = stats['r'].item(), stats['p-val'].item()
        if np.isnan(p): continue

        # Record
        df.loc[df.shape[0]] = [edges[e], heads[h], n_data, corr, p]

    # Add FDR
    df['fdr'] = scipy.stats.false_discovery_control(df['p'].to_numpy())

    # Sort by p/fdr
    df = df.sort_values('p').reset_index(drop=True)

    return df


def get_prs_df(targets, *, meta, data, edges, heads, subject_ids, prs_col='prs_scaled_SCZ.3.5_MVP', **kwargs):
    """
    Get PRS and attention for specified edge-head targets

    `targets` is expected to be a numpy array of shape (-1, 2) which contains
    edge-head target pairs on each row.
    """

    # Get PRS scores
    # NOTE: There are several prs_SCZ columns, which to use is unclear

    prs = meta.set_index('SubID').loc[subject_ids, prs_col]

    # Filter to subjects with PRS scores
    present_id_mask = (~prs.isna()).to_numpy()
    data = data[:, :, present_id_mask]
    subject_ids = np.array(subject_ids)[present_id_mask]
    prs = prs.loc[~prs.isna()].to_numpy()

    # Format df
    df = pd.DataFrame()
    for edge, head in targets:
        # Find idx
        edge_idx = np.argwhere(np.array(edges)==edge)[0][0]
        head_idx = np.argwhere(np.array(heads)==head)[0][0]

        # Get new data
        att = data[edge_idx, head_idx]
        df_new = pd.DataFrame({'Subject': subject_ids, 'Edge': edge, 'Head': head, 'PRS': prs, 'Attention': att}).dropna()
        df = pd.concat((df, df_new))
    # Combined name
    df['Name'] = df.apply(lambda r: f'{r["Edge"]} ({r["Head"]})', axis=1)

    # Annotate high/low prs
    cutoffs = {'low': meta[prs_col].quantile(1/3), 'mid': meta[prs_col].quantile(2/3), 'high': meta[prs_col].quantile(1)}
    def assign_cutoff(x):
        for s, v in cutoffs.items():
            if x < v: break
        return s
    df['Risk'] = df['PRS'].map(assign_cutoff)

    return df
