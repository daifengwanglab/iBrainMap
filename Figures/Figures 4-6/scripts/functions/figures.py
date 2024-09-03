import matplotlib.pyplot as plt

from .computation import *
from .plotting import *


### Figure functions
def figure_regulon_statistics(graph_summary):
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    # Phenotype and Sub-Phenotype vs TF Outgoing
    plot_tf_outgoing(graph_summary, ax=axs[0])
    # Phenotype and Sub-Phenotype vs Closeness
    plot_statistic(graph_summary, ax=axs[1])

    return fig


def figure_diffusion(
        diff_g_individuals,
        diff_graph_summary_coex,
        diff_graph_summary_att,
        *,
        meta,
        individual_sample_ids,
):
    # Generate figure layout
    mosaic = [
        ['A2', 'A2', 'A2', 'B1', 'B1', 'B1',],
        ['A2', 'A2', 'A2', 'B1', 'B1', 'B1',],
        ['A2', 'A2', 'A2', 'B1', 'B1', 'B1',],
        ['.', 'D1', 'D1', 'E1', 'E1', 'E1',],
        ['.', 'D1', 'D1', 'E2', 'E2', 'E2',],
    ]
    fig, axs = get_mosaic(mosaic)
    axs['E1'].get_shared_x_axes().join(axs['E1'], axs['E2'])

    # Get common pos for `diff_g_individual` and `diff_g_other`
    concatenated_graph = concatenate_graphs(*diff_g_individuals)
    concatenated_pos = get_graph_pos(concatenated_graph)

    # Novel personal graph
    ax = axs['A2']
    plot_label('A', ax=ax)
    pos = convert_vertex_map(concatenated_graph, diff_g_individuals[0], concatenated_pos)
    visualize_graph(diff_g_individuals[0], pos=pos, ax=ax)
    ax.axis('off')
    ax.set_title('Personal Subgraph')

    # Other sample personal graph
    ax = axs['B1']
    plot_label('B', ax=ax)
    pos = convert_vertex_map(concatenated_graph, diff_g_individuals[1], concatenated_pos)
    visualize_graph(diff_g_individuals[1], pos=pos, ax=ax)
    ax.axis('off')
    ax.get_legend().set_visible(False)
    ax.set_title('Alternative Personal Subgraph')

    # NPS comparison
    ax = axs['D1']
    plot_label('C', ax=ax)
    plot_nps(meta, individual_sample_ids, ax=ax)
    ax.set_title('NPS Comparison')

    # TF closeness coex
    ax = axs['E1']
    plot_label('D', ax=ax)
    plot_statistic(diff_graph_summary_coex, ax=ax)
    ax.set_title('Average TF Closeness')
    ax.set_ylabel('Coexpression')
    ax.set_xlabel(None)
    ax.set_xticklabels([])

    # TF closeness att
    ax = axs['E2']
    plot_statistic(diff_graph_summary_att, ax=ax)
    ax.get_legend().set_visible(False)
    ax.set_ylabel('Attention')

    return fig


def figure_data_driven(
        data_g_individuals,
        diff_graph_summary,
        data_graph_summary,
        *,
        meta,
        individual_sample_ids,
):
    # Generate figure layout
    mosaic = [
        ['A1', 'A1', 'A2', 'A2', 'B1', 'B1',],
        ['A1', 'A1', 'A2', 'A2', 'B2', 'B2',],
        ['A3', 'A3', 'A4', 'A4', 'C1', 'C1',],
        ['A3', 'A3', 'A4', 'A4', 'C1', 'C1',],
    ]
    fig, axs = get_mosaic(mosaic)
    axs['B1'].get_shared_x_axes().join(axs['B1'], axs['B2'])

    # Get common pos for `diff_g_individual` and `data_g_other`
    concatenated_graph = concatenate_graphs(*data_g_individuals)
    concatenated_pos = get_graph_pos(concatenated_graph)

    ### Personal Graphs
    for i, (graph, name) in enumerate(zip(data_g_individuals, individual_sample_ids)):
        ax = axs[f'A{i+1}']
        if i==0: plot_label('A', ax=ax)
        pos = convert_vertex_map(concatenated_graph, graph, concatenated_pos)
        visualize_graph(graph, pos=pos, ax=ax)
        ax.axis('off')
        if i!=0: ax.get_legend().set_visible(False)
        ax.set_title(name)

    # TF closeness diff
    ax = axs['B1']
    plot_label('B', ax=ax)
    plot_statistic(diff_graph_summary, col='Cliques', ax=ax)
    ax.set_ylabel('Diffusion')
    ax.set_xlabel(None)
    ax.set_xticklabels([])

    # TF closeness data
    ax = axs['B2']
    plot_statistic(data_graph_summary, col='Cliques', ax=ax)
    ax.set_title('Number of Cliques')
    ax.get_legend().set_visible(False)
    ax.set_ylabel('Data Driven')

    # NPS Plot
    ax = axs['C1']
    plot_label('C', ax=ax)
    plot_nps(meta, individual_sample_ids, ax=ax)
    ax.set_title('NPS Comparison')

    return fig
