import colorsys
import re
import string
import textwrap
import warnings

import graph_tool.all as gt
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from .file import *


EDGE_SPLIT_STRING = ' --> '


### Utility functions
def rgba_to_hex(rgba):
    # No alpha included
    int_rgba = [int(255*i) for i in rgba]
    return '#{:02x}{:02x}{:02x}'.format(*int_rgba)


def rgba_array_to_rgba_string(array):
    int_rgba = [int(255*i) for i in array[:3]] + [array[3]]
    return 'rgba({:d},{:d},{:d},{:f})'.format(*int_rgba)


def combine_graphs(graphs, dynamic_load=True, **kwargs):
    # Merge graphs
    graph = None
    graphs_len = []
    for i, graph_new in enumerate(graphs):
        if dynamic_load:
            try:
                graph_new =  load_graph_by_id(graph_new, **kwargs)
            except:
                continue
        graph_new = graph_new.rename(columns={'coef': f'coef_{i}'})
        graphs_len.append(len(graph_new))
        if graph is None:
            graph = graph_new
        else:
            graph = graph.merge(
                graph_new,
                how='outer',
                on=['TF', 'TG'],
            )

    # Raise exception if no graphs
    if graph is None:
        raise LookupError('No graphs found.')

    # Replace nan with 0
    graph = graph.fillna(0)

    # Average new coefs
    graph['coef'] = graph.iloc[:, 2:].mean(axis=1)

    # Match size to originals
    # graphs_avg_len = np.mean(graphs_len)
    # graph = graph.sort_values('coef')
    # graph = graph.iloc[:int(graphs_avg_len)]

    # Cull edges for high coef
    # graph = graph[graph['coef'] > graph['coef'].quantile(.8)]

    return graph[['TF', 'TG', 'coef']]


def simulate_diffusion(g, genes, spread=.1, eps=1e-6, color_power=1):
    # Perform diffusion recursively
    g.vp.diffusion_value = g.new_vertex_property('double')
    vertices = []
    for v in g.vertices():
        if g.vp.ids[v] in genes:
            g.vp.diffusion_value[v] = 1
            # Both out and in neighbors
            vertices.append(g.get_out_neighbors(v))
            vertices.append(g.get_in_neighbors(v))
    if len(vertices) > 0:
        vertices = np.concatenate(vertices)
        g = simulate_diffusion_helper(g, vertices, 1, spread, eps)

    # Convert to color
    g.vp.diffusion_color = g.new_vertex_property('string')
    for v in g.vertices():
        # Red to white as it spreads
        val = 1-g.vp.diffusion_value[v]**color_power
        g.vp.diffusion_color[v] = rgba_to_hex((1, val, val, 1))

    return g


def simulate_diffusion_helper(g, vertices, value, spread, eps):
    if value < eps or len(vertices) == 0:
        return g
    new_value = spread * value
    new_vertices = []
    for v in vertices:
        if new_value > g.vp.diffusion_value[v]:
            g.vp.diffusion_value[v] = new_value
            new_vertices.append(g.get_out_neighbors(v))
            new_vertices.append(g.get_in_neighbors(v))
    if len(new_vertices) == 0: return g
    new_vertices = np.concatenate(new_vertices)

    return simulate_diffusion_helper(g, new_vertices, new_value, spread, eps)


def detect_synthetic_vertices_list(graph, hub_present=False):
    if hub_present:
        synthetic_vertices = list(graph[graph['TG'] == 'hub']['TF'])
    else:
        synthetic_vertices = [
            s for s in np.unique(list(graph['TG']) + list(graph['TF']))
            if string_is_synthetic(s)]
        synthetic_vertices += ['EN', 'IN', 'OPC', 'PC']
    return synthetic_vertices


def string_is_synthetic(s):
    "Detect synthetic node ids"
    if (
        (s != s.upper() and not re.compile('^C\d.+$').match(s))
        or s in ['EN', 'IN', 'OPC', 'PC', 'VLMC', 'PVM', 'SMC']
        or sum([s.startswith(t) for t in ['EN_', 'IN_', 'CD8_']])
    ): return True
    return False


def detect_synthetic_vertices_graph(g):
    # If has g.vp.ids
    return [g.vp.ids[v] for v in g.vertices() if string_is_synthetic(g.vp.ids[v])]

    # If has g.vp.text_synthetic[v]
    # return [g.vp.ids[v] for v in g.vertices() if g.vp.text_synthetic[v]]


def subset_graph(source, target):
    "Subset `source` graph nodes to those present in `target`"
    return gt.GraphView(
        source,
        vfilt=[source.vp.ids[v] in [target.vp.ids[v]
                                    for v in target.vertices()]
                for v in source.vertices()],
    )


def subset_by_hub(g, vertices, verticies_are_ids=True, include_synthetic=False):
    """
    Subsets a graph to nodes in `g` connected to those in `vertices`.
    """
    # TODO: Add degrees of difference, i.e. 2nd gen connections and below
    # Convert ids to vertices
    vertices_new = []
    if verticies_are_ids:
        for vid in vertices:
            vertices_new.append(gt.find_vertex(g, g.vp.ids, vid)[0])
    vertices = vertices_new

    # Select vertices
    vfilt = []
    for v in vertices:
        vfilt.append(v)
        for vn in v.all_neighbors():
            if include_synthetic or not g.vp.text_synthetic[vn]:
                vfilt.append(vn)

    # Convert to proper format
    vfilt = [v in vfilt for v in g.vertices()]

    return gt.GraphView(
        g,
        vfilt=vfilt,
    )


def concatenate_graphs(
        *graphs,
        color_by_source=True,
        recolor=True,
        exclude_zeroes_from_mean=True,
        remove_duplicate_edge=True,
        recalculate='After',  # Can be 'Before', 'After', or False
        threshold=None):
    "Concatenate all graphs provided, assesses duplicates by IDs"
    # Default parameters
    if threshold is None or threshold == True:
        threshold = len(graphs)**(1/3 - 1)
        print(f'No threshold provided, using threshold of {threshold}.')
        # threshold = np.log(1+len(graphs)) / len(graphs)

    # Start aggregating
    g = None
    g_coefs = {}
    for i, gc in enumerate(graphs):
        # Copy if first
        if not g:
            _add_attribute_to_dict(
                g_coefs,
                gc.edges(),
                indexer=lambda e: get_edge_string(gc, e),
                attribute=gc.ep.coef,
                default=lambda: [0 for _ in range(len(graphs))],
                index=i,
            )
            # Convert view to graph
            # NOTE: Union does not play nicely with views in the first argument,
            #   hence, why we have to do this on the first graph
            g = gt.graph_union(gt.Graph(), gc, include=True, internal_props=True)
            continue

        # Get common vertices
        vertex_map = gc.new_vertex_property('int')
        for v in gc.vertices():
            # Find corresponding v in g
            vid = gc.vp.ids[v]
            idx = gt.find_vertex(g, g.vp.ids, vid)  # Not really idx, actually list of vertex refs

            # Cases for finding
            if not idx:
                vertex_map[v] = -1
            elif len(idx) == 1:
                vertex_map[v] = int(idx[0])
            else:
                raise LookupError(f'ID \'{vid}\' has duplicate entries in \'g\'.')


        # Track coefs
        _add_attribute_to_dict(
            g_coefs,
            gc.edges(),
            indexer=lambda e: get_edge_string(gc, e),
            attribute=gc.ep.coef,
            default=lambda: [0 for _ in range(len(graphs))],
            index=i,
        )

        # Concatenate (assumes all vertex and edge properties are the same)
        # TODO: Other props, add coef averaging
        # g, props = gt.graph_union(
        g = gt.graph_union(
            g,
            gc,
            intersection=vertex_map,
            include=True,
            internal_props=True)

    # Label self loops
    g.ep.self_loop = g.new_edge_property('bool')
    gt.label_self_loops(g, eprop=g.ep.self_loop)

    # Remove duplicate edges
    if remove_duplicate_edge:
        g = remove_duplicate_edges(g)

    # Add processed attributes
    g.vp.self_loop_value = g.new_vertex_property('double')
    # g.ep.color = g.new_edge_property('vector<double>')
    g.ep.coefs = g.new_edge_property('vector<double>')
    for e in g.edges():
        # Get coefs
        coefs = g_coefs[get_edge_string(g, e)]
        g.ep.coefs[e] = coefs

        # Get processed attributes
        in_graph = [c != 0 for c in coefs]
        present_coef = np.mean([c for c in coefs if c != 0]) if exclude_zeroes_from_mean else np.mean(coefs)
        color = [0 for _ in range(4)]

        # Set color
        if color_by_source:
            color[:3] = _determine_color(g, e)

        # Set alpha
        color[3] = get_alpha(present_coef) / len(graphs)**(1/4)

        # Write
        if recolor:
            g.ep.color[e] = color
        if g.ep.self_loop:
            g.vp.self_loop_value[e.source()] = present_coef

    # Reassign TF-TG text/shape/color
    # NOTE: This can be calculated before edge filtering for more informative coloration,
    #  this might not match exactly with the output graph edges, but instead implies
    #  that SOME combination of graphs found node X to be of type Y
    if recalculate == 'Before':
        g = assign_vertex_properties(g)

    # Remove infrequent edges
    if threshold:
        num_vertices = g.num_vertices()
        num_edges = g.num_edges()
        g = filter_to_common_edges(g, threshold=threshold)
        g = cull_isolated_leaves(g)
        print(
            f'Filtered from {num_vertices} vertices and {num_edges} edges '
            f'to {g.num_vertices()} vertices and {g.num_edges()} edges via '
            f'common edge filtering.')

    # Reassign TF-TG text/shape/color
    if recalculate == 'After':
        g = assign_vertex_properties(g)

    return g


def _determine_color(g, e, method='presence'):
    coefs = g.ep.coefs[e]

    # Get color index
    if method == 'presence':
        # Default method assigns binary string to presence
        # i.e. using two: '11' for both, '10' for second, etc.
        cindex = ''.join([str(int(b)) for b in [c != 0 for c in coefs]])[::-1]
    elif method == 'max':
        # Only assign '1' to max entry
        cindex = '0' * len(coefs)
        cindex = list(cindex)  # Dumb python string indexing workaround
        cindex[-np.argmax(coefs)] = '1'
        cindex = ''.join(cindex)
    else:
        raise AttributeError(f'Method {method} not found.')

    # Convert to color
    cindex = int(cindex, 2) # Binary to int
    if cindex == 2**len(coefs) - 1: return [0 for _ in range(3)]
    if len(coefs) == 2:
        # Custom colors
        if cindex == 1:
            hue = 2/3.
        if cindex == 2:
            hue = 0.
        if cindex == 0:  # Not present in any
            hue = 1/3.
    else:
        hue = cindex * (1. / (2**len(coefs) - 2))
    color = colorsys.hsv_to_rgb(hue, 1, .5)

    return color


def get_alpha(coef):
    x = np.log10(1+coef)  # Log scaling
    alpha = x / (1+x)
    alpha = .1 + .2 * alpha  # Add floor and ceiling
    return alpha


def get_edge_string(g=['', ''], e=None):
    # If g is a list
    if type(g) == type([]):
        source_str, target_str = g
    # If g is a graph
    else:
        assert e is not None, 'If a graph is provided, `e` must not be None'
        source_str, target_str = g.vp.ids[e.source()], g.vp.ids[e.target()]

    return f'{source_str}{EDGE_SPLIT_STRING}{target_str}'


def split_edge_string(s):
    return s.split(EDGE_SPLIT_STRING)


def _add_attribute_to_dict(dict, iterator, *, indexer=lambda x: x, attribute, default=lambda: [], index=None):
    """
    Add result `dict[indexer[i]] = attribute[i]` for `i` in `iterator` in-place
    """
    for item in iterator:
        if not indexer(item) in dict:
            dict[indexer(item)] = default()
        if index is None:
            dict[indexer(item)].append(attribute[item])
        else:
            dict[indexer(item)][index] = attribute[item]


def _normalize_dict_item_length(dict, length, default=0):
    """
    Normalize dictionary array length in-place
    """
    for k in dict.keys():
        if len(dict[k]) > length:
            raise IndexError('Dictionary entry too long.')
        elif len(dict[k]) < length:
            dict[k] += [default for _ in range(length - len(dict[k]))]


def convert_vertex_map(source_graph, target_graph, vertex_map):
    "Convert pos to another graph"
    # NOTE: Probably a way to do this without `source_graph`
    # SECOND NOTE: Probably not
    # Debug
    # tg = [target_graph.vp.ids[v] for v in target_graph.vertices()]
    # sg = [source_graph.vp.ids[v] for v in source_graph.vertices()]
    # tg_in_sg = [vid in sg for vid in tg]
    # print(sum(tg_in_sg) / len(tg))  # Should be 1.0 if all values of `target_graph` are in `source_graph`

    # Find corresponding values and record
    converted_map = target_graph.new_vertex_property('vector<double>')
    for v in source_graph.vertices():
        # Find corresponding v in target_graph
        vid = source_graph.vp.ids[v]
        idx = gt.find_vertex(target_graph, target_graph.vp.ids, vid)

        # Cases for finding
        if idx:
            if len(idx) > 1: raise LookupError(f'ID \'{vid}\' has duplicate entries in \'g\'.')
            converted_map[idx[0]] = vertex_map[v]

    # Debug
    # for v in target_graph.vertices():
    #     if not converted_map[v]:
    #         print(f'\'{target_graph.vp.ids[v]}\' not defined in `vertex_map`.')

    return converted_map


def remove_text(g, preserve_synthetic=False):
    # Get no synthetic view
    if preserve_synthetic:
        g_view = gt.GraphView(
            g,
            vfilt=lambda v: g.vp.text_synthetic[v] == '',
        )
    else: g_view = g

    # Remove text
    for v in g_view.vertices():
        g_view.vp.text[v] = ''

    return g


def remove_text_by_centrality(g, preserve_synthetic=True, percentile=90, eps=1e-10):
    # Calculate betweenness
    # vertex_centrality, _ = gt.betweenness(g)
    vertex_centrality = gt.pagerank(g)

    # Get no synthetic view
    if preserve_synthetic:
        g_view = gt.GraphView(
            g,
            vfilt=lambda v: g.vp.text_synthetic[v] == '',
        )
    else: g_view = g
    threshold = np.percentile([vertex_centrality[v] for v in g_view.vertices()], percentile)
    if percentile:
        threshold = max(eps, threshold)  # Use eps as min threshold
    # print(sum([vertex_centrality[v] >= threshold for v in g_view.vertices()]))
    # print([vertex_centrality[v] for v in g_view.vertices()])
    # print(threshold)

    # Remove text
    for v in g_view.vertices():
        if vertex_centrality[v] < threshold:
            g_view.vp.text[v] = ''

    return g


def get_intersection(g):
    return gt.GraphView(
        g,
        efilt=lambda e: sum([val > 0 for val in g.ep.coefs[e]]) == len(g.ep.coefs[e]),
    )


def cull_isolated_leaves(g):
    # Extract largest component
    return gt.extract_largest_component(g, directed=False)

    # # Return maximal clique
    # largest_clique = []
    # for clique in gt.max_cliques(g):
    #     if len(clique) > len(largest_clique):
    #         largest_clique = clique
    # print(f'Largest Clique: {len(largest_clique)}')

    # return gt.GraphView(g, vfilt=lambda v: int(v) in list(largest_clique))

    # # Remove nodes which aren't connected to a synthetic node
    # return gt.GraphView(
    #     g,
    #     vfilt=lambda v: (g.vp.text_synthetic[v] != '') or (len([n for n in v.all_neighbors() if g.vp.text_synthetic[n]]) > 0),
    # )


def remove_duplicate_edges(g):
    # Detect duplicates
    is_duplicate = g.new_edge_property('bool')
    seen = {}
    print('Removing duplicate edges...')
    for e in tqdm(g.edges(), total=g.num_edges()):
        hashObject = (int(e.source()), int(e.target()))
        if hashObject in seen: is_duplicate[e] = True
        else:
            is_duplicate[e] = False
            seen[hashObject] = True

    # Filter
    return gt.GraphView(
        g,
        efilt=lambda e: not is_duplicate[e],
    )


def _is_duplicate_edge(g, e):
    # Naive duplicate edge detection
    source = e.source()
    target = e.target()

    # Detect matches
    for f in g.edges():
        if e == f:
            # Return false if duplicate but first instance
            return False
        if source == f.source() and target == f.target():
            return True
    return False


def compute_differences(g):
    # Compute range for each edge coef
    g.ep.coef_diff = g.new_edge_property('double')
    for e in g.edges():
        coefs = [val for val in g.ep.coefs[e] if val]  # Only use nonzero values
        g.ep.coef_diff[e] = max(coefs) - min(coefs)

    return g


def color_by_significance(g):
    # Get diff if not present
    try:
        g.ep.coef_diff
    except:
        compute_differences(g)

    # Set color
    for e in g.edges():
        color = [0 for _ in range(4)]
        color[:3] = _determine_color(g, e, method='max')  # Color
        color[3] = get_alpha(g.ep.coef_diff[e])  # Opacity based on diff
        g.ep.color[e] = color

    return g


def get_default_scale(g):
    return .01 * (1 + np.log(g.num_vertices()))


def transfer_text_labels(g, gc):
    "Transfer text labels from `g` to `gc` inplace."
    for v in gc.vertices():
        matches = gt.find_vertex(g, g.vp.ids, gc.vp.ids[v])
        if matches: gc.vp.text[v] = g.vp.text[matches[0]]

    return gc


def scale_edge_coefs(g, scale):
    "Scale edge coefs in `g` by `scale`."
    for e in g.edges():
        g.ep.coef[e] *= scale

    return g


def scale_edge_coefs_list(graph, scale):
    "Scale edge coefs in `g` by `scale`."
    graph['coef'] *= scale

    return graph


def filter_to_common_edges(g, threshold=.6):
    "Filter concatenated graph `g` to edges common among `threshold` of generating graphs."
    # Find edges under threshold
    to_remove = []
    for e in g.edges():
        coefs = g.ep.coefs[e]
        if len(coefs) < 1:
            print('Edge found with no coefs.')
        present = sum([c!=0 for c in coefs]) / len(coefs)
        if present < threshold:
            to_remove.append(e)

    # Remove matching edges
    for e in to_remove: g.remove_edge(e)

    return g


def get_inverse_graph(g, subgraph):
    "Get all nodes and edges in `g` but not in `subgraph`"
    # Find edges and vertices to remove
    vertices_to_remove = []; edges_to_remove = []
    for v in g.vertices():
        query = gt.find_vertex(subgraph, subgraph.vp.ids, g.vp.ids[v])
        if len(query) > 0: vertices_to_remove.append(v)
    subgraph_edges = [get_edge_string(subgraph, e) for e in subgraph.edges()]  # Naive method, vulnerable to double-hyphen ids
    for e in g.edges():
        query = get_edge_string(g, e) in subgraph_edges
        if query: edges_to_remove.append(e)

    # Remove matching vertices and edges
    # NOTE: Actually removing vertices invalidates descriptors and vertex properties
    # for e in edges_to_remove: g.remove_edge(e)  # Need to remove edges first because of below
    # g.remove_vertex(vertices_to_remove)  # NOTE: Must be done this way or removed in descending order

    return gt.GraphView(
        g,
        vfilt=lambda v: v not in vertices_to_remove,
        efilt=lambda e: e not in edges_to_remove,
    )


def remove_edges(g):
    "Remove all edges from `g`"
    edges = [e for e in g.edges()]
    for e in edges: g.remove_edge(e)

    return g


def get_node_appearance(node_type=None):
    "Returns color for corresponding node type"
    # Parameters
    # sizes = [1.2, 1., .75, .5]
    sizes = np.array(list(range(5))[::-1])
    sizes = (sizes - sizes.min()) / (sizes.max() - sizes.min())
    sizes = sizes * .8 + .1
    # palette = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Get correct values
    if node_type is None:
        color = '#FFFFFF'
        shape = 'circle'  # Default
        size = sizes[3]
    elif node_type.upper() == 'HUB':
        # color = rgba_to_hex(palette[0])
        color = '#000000'
        shape = 'octagon'
        size = sizes[0]
    elif node_type.upper() == 'CELLTYPE':
        # g.vp.color[v] = rgba_to_hex(palette[1])
        color = '#65CA5C'
        shape = 'hexagon'
        size = sizes[1]
    elif node_type.upper() == 'TFTG':
        # g.vp.color[v] = rgba_to_hex(palette[4])
        # color = '#47653A'
        color = '#A3D83E'
        shape = 'triangle'  # Same as just tf
        size = sizes[2]
    elif node_type.upper() == 'TF':
        # g.vp.color[v] = rgba_to_hex(palette[2])
        color = '#A3D83E'
        shape = 'triangle'
        size = sizes[2]
    elif node_type.upper() == 'TG':
        # g.vp.color[v] = rgba_to_hex(palette[3])
        color = '#A7C69D'
        shape = 'circle'
        size = sizes[3]
    elif node_type.upper() == 'SNP':
        color = '#000000'
        shape = 'circle'
        size = sizes[4]
    else:
        raise ValueError(f'No node type \'{node_type}\' found.')

    return color, shape, size


def assign_vertex_properties(g):
    # Detect synthetic vertices
    synthetic_vertices = detect_synthetic_vertices_graph(g)

    # View without synthetic nodes or self loops
    # Need to do `vfilt` slowly bc `g.vp.ids.fa` doesn't work with string
    # DO NOT use [... for ... in g.vertices/edges()] as the ordering is not the same
    g_nosynthetic = gt.GraphView(
        g,
        vfilt=lambda v: g.vp.ids[v] not in synthetic_vertices,
        efilt=lambda e: not g.ep.self_loop[e],
    )

    g.vp.color = g.new_vertex_property('string')  # Can't show with text if set to `vector<double>``
    g.vp.shape = g.new_vertex_property('string')
    g.vp.size = g.new_vertex_property('double')
    g.vp.text_synthetic = g.new_vertex_property('string')
    g.vp.text = g.new_vertex_property('string')
    g.vp.node_type = g.new_vertex_property('string')
    for v in g.vertices():
        # assert g.vp.ids[v] == g_nosynthetic.vp.ids[v]  # Works!
        v_id = g.vp.ids[v]
        # Hub
        if v_id in ['hub']:
            g.vp.node_type[v] = 'hub'
            g.vp.color[v], g.vp.shape[v], g.vp.size[v] = get_node_appearance(g.vp.node_type[v])
            g.vp.text_synthetic[v] = v_id
            g.vp.text[v] = v_id
            # root = v
        # Cell-type
        elif v_id in synthetic_vertices:  # NOTE: This does not include 'hub'
            g.vp.node_type[v] = 'celltype'
            g.vp.color[v], g.vp.shape[v], g.vp.size[v] = get_node_appearance(g.vp.node_type[v])
            g.vp.text_synthetic[v] = v_id
            g.vp.text[v] = v_id
        # Default
        else:
            # Regular criterion
            # is_tf = g_nosynthetic.get_out_degrees([v])[0] > 0
            # is_tg = g_nosynthetic.get_in_degrees([v])[0] > 0

            # Synthetic criterion
            neighbors = list(v.all_neighbors())
            is_tf = np.any([string_is_synthetic(g.vp.ids[n]) for n in neighbors])
            second_degrees = sum([list(m.all_neighbors()) for m in neighbors if not string_is_synthetic(g.vp.ids[m])], [])
            is_tg = np.any([string_is_synthetic(g.vp.ids[n]) for n in second_degrees])
            # Debugging
            # print(g.vp.ids[v])
            # for w in second_degrees:
            #     if string_is_synthetic(g.vp.ids[w]):
            #         print(g.vp.ids[w])
            # print()

            # Color
            if is_tf and is_tg:
                g.vp.node_type[v] = 'tftg'
                g.vp.color[v], g.vp.shape[v], g.vp.size[v] = get_node_appearance(g.vp.node_type[v])
            elif is_tf and not is_tg:
                g.vp.node_type[v] = 'tf'
                g.vp.color[v], g.vp.shape[v], g.vp.size[v] = get_node_appearance(g.vp.node_type[v])
            elif not is_tf and is_tg:
                g.vp.node_type[v] = 'tg'
                g.vp.color[v], g.vp.shape[v], g.vp.size[v] = get_node_appearance(g.vp.node_type[v])
            else:
                # Only connections from synthetic node
                g.vp.color[v], g.vp.shape[v], g.vp.size[v] = get_node_appearance()

            # Add text to outer nodes (optional)
            g.vp.text[v] = v_id

    return g


def make_vertices_white(g):
    for v in g.vertices():
        g.vp.color[v] = '#FFFFFF'

    return g


def has_duplicate_vertex_ids(g):
    seen = []
    for v in g.vertices():
        if g.vp.ids[v] in seen: return True
        seen.append(g.vp.ids[v])

    return False


def scale_pos_to_range(g, pos, box_size=2):
    "Scale `pos` vertex attribute to reasonable range, used to combat auto-scaling"
    # Get ranges
    max_values = [-np.inf, -np.inf]
    min_values = [+np.inf, +np.inf]
    for i, v in enumerate(g.vertices()):
        for j, val in enumerate(pos[v]):
            if val < min_values[j]:
                min_values[j] = val
            if val > max_values[j]:
                max_values[j] = val
    num_vertices = i

    # Calculate box size
    if box_size is None:
        box_size = num_vertices**(1/2)

    # Set new sizes
    for v in g.vertices():
        pos[v] = [
            box_size * (pos[v][j] - min_values[j]) / (max_values[j] - min_values[j]) - (box_size / 2)
            for j in range(2)
        ]

    return pos


def scale_pos_by_distance(g, pos, exponent=10):
    "Scale pos by location from center"
    for i, v in enumerate(g.vertices()):
        # Pure transform
        x, y = pos[v]
        distance = (x**2 + y**2)**(1/2)
        angle = np.arctan2(y, x)  # `arctan2` keeps quadrants in mind

        # Get new distance
        if exponent < 1:
            new_distance = distance**exponent
        else:
            new_distance = np.log(1+distance) / np.log(exponent)

        x = new_distance * np.cos(angle)
        y = new_distance * np.sin(angle)
        pos[v] = [x, y]

        # Square transform
        # pos[v] = [np.sign(x) * np.abs(x)**exponent for x in pos[v]]

    return pos


def filter_graph_by_synthetic_vertices(g, *, vertex_ids, max_tfs=-1, max_tgs=-1):
    """
    Filter graph by synthetic vertices `vertex_ids` with `max_tf` tfs and `max_tf` tgs

    `max_tfs` and `max_tgs`: -1 indicates unlimited
    """
    # TODO: Filter recursively rather than only two layers deep
    # Check that all provided vertices are synthetic
    assert np.array([string_is_synthetic(s) for s in vertex_ids]).all()

    # Setup
    vertices_to_keep = []

    # Detect synthetic nodes and propagate
    for v in g.vertices():
        ## Record synthetic nodes
        if g.vp.ids[v] not in vertex_ids:  # Only continue if specified synthetic node
            continue
        vertices_to_keep += [v for v in g.vertices() if g.vp.ids[v] in vertex_ids]

        ## Record top TFs
        # Get all TFs
        tf_edges = list(v.in_edges())
        tf_edges = np.array([e for e in tf_edges if not string_is_synthetic(g.vp.ids[e.source()])])

        # Rank order
        weights = [g.ep.coef[e] for e in tf_edges]
        top_tfs = [e.source() for e in tf_edges[np.argsort(weights)[::-1][:max_tfs]]]

        # Record
        vertices_to_keep += top_tfs

        ## Record top TGs
        for w in top_tfs:
            # Get all TGs
            tg_edges = list(w.out_edges())
            tg_edges = np.array([e for e in tg_edges if not string_is_synthetic(g.vp.ids[e.source()])])

            # Rank order
            weights = [g.ep.coef[e] for e in tg_edges]
            top_tgs = [e.source() for e in tg_edges[np.argsort(weights)[::-1][:max_tgs]]]

            # Record
            vertices_to_keep += top_tgs

    # Filter
    g = gt.GraphView(
        g,
        vfilt=lambda v: v in vertices_to_keep
    )

    return cull_isolated_leaves(g)


def get_all_synthetic_vertices(g):
    return [g.vp.ids[v] for v in g.vertices() if string_is_synthetic(g.vp.ids[v])]


def filter_to_synthetic_vertices(g, vertex_ids=None, depth=2, limit=5):
    # Default
    if vertex_ids is None: vertex_ids = get_all_synthetic_vertices(g)

    # Check that all provided vertices are synthetic
    assert np.array([string_is_synthetic(s) for s in vertex_ids]).all()

    # Detect synthetic nodes
    synthetic_nodes = []
    for v in g.vertices():
        # Record synthetic nodes
        if g.vp.ids[v] not in vertex_ids:  # Only continue if specified synthetic node
            continue
        synthetic_nodes += [v for v in g.vertices() if g.vp.ids[v] in vertex_ids]

    # Recursively propagate
    layers = [synthetic_nodes]
    synthetic = True
    visited = {v: 0 for v in layers[-1]}
    while depth:
        # Get last layer
        last_layer = layers[-1]

        # Get neighbors
        if synthetic:
            # current_layer = [list(v.in_neighbors()) for v in last_layer]  # When CT-TF edges are reversed
            current_layer = [list(v.out_neighbors()) for v in last_layer]
        else:
            current_layer = [list(v.out_neighbors()) for v in last_layer]

        # Exclude synthetic nodes
        current_layer = [[v for v in l if not string_is_synthetic(g.vp.ids[v])] for l in current_layer]

        # Clean list
        def hash_and_return(v): visited[v] = 0; return v
        current_layer = [[hash_and_return(v) for v in l if v not in visited] for l in current_layer]

        # Filter
        if limit != -1:
            attentions = [
                # [g.edge(v, w) if not synthetic else g.edge(w, v) for w in current_layer[i]]  # When CT-TF edges are reversed
                [g.edge(v, w) for w in current_layer[i]]
                for i, v in enumerate(last_layer)
            ]
            to_keep = [np.argsort(att)[-limit:] for att in attentions]
            current_layer = [list(np.array(l)[k]) for l, k in zip(current_layer, to_keep)]

        # Flatten
        current_layer = sum(current_layer, [])

        # Append
        layers.append(current_layer)

        # Iterate
        synthetic = False
        depth -= 1

    # Format
    vertices_to_keep = sum(layers, [])

    # Filter
    g = gt.GraphView(
        g,
        vfilt=lambda v: v in vertices_to_keep
    )

    return cull_isolated_leaves(g)


def join_df_subgroup(
        df_subgroup,
        column='Variance',
        sort_column='Difference--Population',
        num_sort=20,
        filter_to_common=True):
    "Join all keys of `df_subgroup`"
    # Construct
    df = pd.DataFrame()
    for key, df_sub in df_subgroup.items():
        # Destructive
        df_sub.index = df_sub['Edge']
        df_sub = df_sub.rename(columns={column: key})
        df = df.join(df_sub[[key]], how='outer')

    # Filter to edges included in all contrasts
    if filter_to_common:
        df = df.dropna()

    # Sort
    sort_methods = sort_column.split('--')
    for sort_column in sort_methods:
        if sort_column == 'Difference':
            df['Difference'] = np.var(df[[c for c in df.columns if c != 'Population']].to_numpy(), axis=1)
            df = df.sort_values(by=sort_column, ascending=False)
            df = df[[c for c in df.columns if c != 'Difference']]
            if len(sort_methods) > 1:
                df = df.iloc[:num_sort]
        elif sort_column:
            df = df.sort_values(by=sort_column, ascending=False)

    return df


def get_many_graph_lists(subject_ids, columns, join_type='inner'):
    # TODO: Implement for more than two graph lists
    assert len(subject_ids) == 2, '`get_many_graph_lists` can only join 2 graph lists at the moment'

    # Get graphs
    graphs = []
    for sid in subject_ids:
        ## Format and append graph DataFrames
        # Load graph
        graph = load_graph_by_id(sid, column=columns)
        # Remove self loops
        graph = graph.loc[graph.apply(lambda d: d['TF'] != d['TG'], axis=1)]
        # Apply edge strings
        graph['Edge'] = graph.apply(lambda d: get_edge_string([d['TF'], d['TG']]), axis=1)
        graph = graph.drop(columns=['TF', 'TG'])
        # Format
        graph.set_index('Edge')
        # Record
        graphs.append(graph)

    # Join graphs
    graph_1, graph_2 = graphs
    joined_graphs = graph_1.join(graph_2, how=join_type, lsuffix='_s1', rsuffix='_s2')  # Only take edge intersections
    joined_graphs = joined_graphs.drop(columns='Edge_s2').rename(columns={'Edge_s1': 'Edge'}).set_index('Edge')

    return joined_graphs


def get_top_idx(df, columns, num_edges_per_head=2):
    "Return top `num_edges_per_head` idx in df for each column in `columns` with NO OVERLAP, aggregated"
    # Get greatest idx for each
    idx_to_include = []
    for column in columns:
        # Get *new* top edges for each head
        top_edges = df[column].argsort()[::-1]
        top_edges = list([idx for idx in top_edges if idx not in idx_to_include][:num_edges_per_head])
        idx_to_include += top_edges

    return idx_to_include


def format_return(ret):
    if len(ret) == 1:
        return ret[0]
    return ret


def get_outlier_idx(data, num_std_from_mean=3, return_mask=False):
    "Get outliers (`num_std_from_mean` standard deviations from mean) from `data`"
    data = np.array(data)
    mean = np.mean(data)
    std = np.std(data)
    lower, upper = mean - num_std_from_mean * std, mean + num_std_from_mean * std

    mask = (data <= lower) + (data >= upper)
    idx = np.argwhere(mask)

    ret = (idx,)
    if return_mask: ret += (mask,)
    return format_return(ret)


def get_genomic_coordinates(snp_ids):
    "Get genomic coordinate(s) of `snp_ids`.  If str, will return str.  Otherwise, list"
    # Parameters
    string_return = type(snp_ids) == str
    if string_return:
        snp_ids = [snp_ids]

    # Get coordinates
    # TODO: Check if correct
    # chromosome, coord, first, second = snp_id.split(':')
    ret = []
    for snp_id in snp_ids:
        separated = snp_id.split(':')
        assert len(separated) == 4, print(f'Entry {snp_id} has incorrect format')
        separated[1] = int(separated[1])
        ret.append(separated)

    # Return
    if string_return:
        return ret[0]
    return ret


def get_chromosome_lengths():
    return {
        '1': 247_249_719,
        '2': 242_951_149,
        '3': 199_501_827,
        '4': 191_273_063,
        '5': 180_857_866,
        '6': 170_899_992,
        '7': 158_821_424,
        '8': 146_274_826,
        '9': 140_273_252,
        '10': 135_374_737,
        '11': 134_452_384,
        '12': 132_349_534,
        '13': 114_142_980,
        '14': 106_368_585,
        '15': 100_338_915,
        '16': 88_827_254,
        '17': 78_774_742,
        '18': 76_117_153,
        '19': 63_811_651,
        '20': 62_435_964,
        '21': 46_944_323,
        '22': 49_691_432,
        'X': 154_913_754,
        'Y': 57_772_954,
    }


def get_chromosome_length(chromosome):
    "Return chromosome length"
    length_dictionary = get_chromosome_lengths()

    return length_dictionary[chromosome]


def get_chromosome_order():
    return [
        '1',
        '2',
        '3',
        '4',
        '5',
        '6',
        '7',
        '8',
        '9',
        '10',
        '11',
        '12',
        '13',
        '14',
        '15',
        '16',
        '17',
        '18',
        '19',
        '20',
        '21',
        '22',
        'X',
        'Y',
    ]


def get_chromosome_coordinate(chromosome):
    "Return start coordinate of chromosome"
    chromosome_order = get_chromosome_order()
    chromosome_order = np.array(chromosome_order)

    # Sum previous chromosome lengths
    idx = np.argwhere(chromosome_order==chromosome)
    assert len(idx) > 0, f'Chromosome {chromosome} not found'
    running_sum = 0
    for chromosome in chromosome_order[:idx[0][0]]:
        running_sum += get_chromosome_length(chromosome)

    return running_sum


def add_snp_to_graph(g, *, snp_id, snp_target, silence_exception=True):
    "Add snp to graph"
    # Get matching ids
    matches = gt.find_vertex(g, g.vp.ids, snp_target)

    # Raise error if not found
    if len(matches) == 0:
        if silence_exception: return g
        raise LookupError(f'No id \'{snp_target}\' found in `g`')

    # Add node
    v_snp = g.add_vertex()
    g.vp.node_type[v_snp] = 'snp'
    g.vp.color[v_snp], g.vp.shape[v_snp], g.vp.size[v_snp] = get_node_appearance(g.vp.node_type[v_snp])
    g.vp.ids[v_snp] = snp_id
    g.vp.text[v_snp] = snp_id

    # Add edge
    e_snp = g.add_edge(v_snp, matches[0])
    g.ep.color[e_snp] = [0, 0, 0, 1]
    g.ep.coef[e_snp] = 1

    # Return
    return g


def remove_snps(g):
    # Filter out nodes which have `node_type` 'SNP'
    to_remove = [v for v in g.vertices() if g.vp.node_type[v].upper() == 'SNP']
    return gt.GraphView(g, vfilt=lambda v: v not in to_remove)


def make_snps_invisible(g):
    # Make all nodes with `node_type` 'SNP' have transparent vertex edges
    # Create parameter
    try: g.vp.hide
    except: g.vp.hide = g.new_vertex_property('bool')

    # Change values
    for v in g.vertices():
        if g.vp.node_type[v].upper() == 'SNP':
            g.vp.hide[v] = True

    # Return
    return g


def convert_dosage_ids_to_subject_ids(dosage, *, meta, inplace=False):
    if not inplace:
        dosage = dosage.copy()

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

    # Remove empty columns
    dosage = dosage.loc[:, [c for c in dosage.columns if c is not None]]

    return dosage


def limit_labels(pl, n=10):
    # Only show *up to* `n` x labelslimit_labels
    num_labels = len(pl.get_xticklabels())
    interval = max(1, int(num_labels/n))
    # positions = []; labels = []
    for i, label in enumerate(pl.get_xticklabels()):
        # This way works if `constrained_layout` can handle the invisible labels
        if i % interval != 0: label.set_visible(False)
        # This removes the labels more permanently
        # if i % interval == 0 or i == (num_labels-1): positions.append(label.get_position()[0]); labels.append(label.get_text())
    # pl.set_xticks(positions, labels)


def get_all_synthetic_ids(g):
    synthetic = []
    for v in g.vertices():
        if string_is_synthetic(g.vp.ids[v]):
            synthetic.append(g.vp.ids[v])
    return synthetic


def hex_to_rgb(s):
    return [int(s[i:i+2], 16) for i in range(1, 6, 2)]


def filter_remove_ct_ct(df, col='id'):
    "Filter to remove ct-ct linkages from a DataFrame `df` containing a column `col` storing edge names"
    return df.loc[~df[col].map(lambda s: string_is_synthetic(split_edge_string(s)[0]) and string_is_synthetic(split_edge_string(s)[1]))]


def filter_remove_ct_x(df, col='id'):
    "Filter to remove ct-* linkages from a DataFrame `df` containing a column `col` storing edge names"
    return df.loc[~df[col].map(lambda s: string_is_synthetic(split_edge_string(s)[0]) or string_is_synthetic(split_edge_string(s)[1]))]


def filter_remove_tf_tg(df, col='id'):
    "Filter to remove tf-tg linkages from a DataFrame `df` containing a column `col` storing edge names"
    return df.loc[~df[col].map(lambda s: not string_is_synthetic(split_edge_string(s)[0]) and not string_is_synthetic(split_edge_string(s)[1]))]


def alphabetize_shape(shape, *, offset=0, return_dict=False, return_offset=False, uppercase=False):
    "Alphabetize shape labels for a mosaic layout, useful for adding new panels to existing layouts without having to manually replace"
    # Get order of appearance
    del_chars = ('\n', '.', ' ')
    shape_cull = shape
    for dc in del_chars:
        shape_cull = shape_cull.replace(dc, '')
    shape_cull = np.array(list(shape_cull))
    _, index = np.unique(shape_cull, return_index=True)
    order = [shape_cull[i] for i in sorted(index)]

    # Get conversion
    base = string.ascii_uppercase if uppercase else string.ascii_lowercase
    labels = base[offset:offset+len(order)]  # Only works up to 26 at the moment
    conversion = {k : v for k, v in zip(order, labels)}

    # Replace characters
    new_shape = ''
    for c in shape:
        new_shape += conversion[c] if c not in del_chars else c

    ret = (new_shape,)
    if return_dict: ret += (conversion,)
    if return_offset: ret += (offset+len(order),)
    return format_return(ret)


def shape_array_from_shape(shape):
    """
    Create an array from a `shape` string of the format
    ```
    shape = '''
        xxx
    '''
    ```
    """
    shape_array = [s.replace(' ', '') for s in shape.split('\n')[1:-1]]
    shape_array = np.array([list(s) for s in shape_array])
    return shape_array


def get_colors_from_values(values, min_val=None, max_val=None, start=(.8, .5, .25), end=(1., .1, .1)):
    # Defaults
    if min_val is None: min_val = min(values)
    if max_val is None: max_val = min(values)
    # Normalize values
    normalized = (values-min_val) / (max_val-min_val)
    normalized[normalized < 0] = 0
    normalized[normalized > 1] = 1
    # Calculate colors
    colors = [v*np.array(end) + (1-v)*np.array(start) for v in normalized]
    return colors


def wrap_text(ticklabels, *, chars=20):
    new_ticklabels = []
    for label in ticklabels:
        text = label.get_text()
        new_text = '\n'.join(textwrap.wrap(text, chars))
        label.set_text(new_text)
        new_ticklabels.append(label)
    return new_ticklabels


def check_ct_edge_specificity():
    """
    Check for duplicate edges with different attention weights.
    If exist, throw error.
    """
    # Get all graphs
    for k in tqdm(get_graphs_pkl()):
        df = get_graphs_pkl()[k]
        df['count'] = 1
        dup = df[['from', 'to', 'count']].groupby(['from', 'to']).sum().reset_index()
        df = df.drop(columns='count')
        dup = dup.loc[dup['count'] > 1]
        dup_id = dup.apply(lambda r: f'{r["from"]}_{r["to"]}', axis=1)

        # Check all duplicates for differing attentions
        unique = (~df.loc[dup_id].drop(columns=['ct_GRN', 'edge_type']).duplicated()).sum()  # 'org_edge_weight',
        assert unique == dup_id.shape[0], f'Found {unique - dup_id.shape[0]} non-redundant duplicated edges across cell types.'
        # Naive
        # for id in dup_id:
        #     df_filt = df.loc[id].drop(columns=['org_edge_weight', 'ct_GRN', 'edge_type'])
        #     duplicates = df_filt.duplicated().sum()
        #     assert duplicates == df_filt.shape[0]-1, f'Found edge with difference across cell-types, {id}'


def filter_go_terms(terms, min_go_num=50_000):
    # Filter to only GO terms
    is_go = [term.startswith('GO:') for term in terms]

    # Filter to deep GO terms
    # TODO: Use *.obo file instead of heuristic
    mask = [ig and (int(term[3:]) >= min_go_num) for ig, term in zip(is_go, terms)]

    return mask


def label_add_arrow(labels, ct=False):
    # This function replaces hyphens with arrows
    for l in labels:
        text = l.get_text()
        text = text.split('_')
        if ct: text = '_'.join(text[:-2]) + ': ' + ' --> '.join(text[-2:])
        else: text = ' --> '.join(text)
        l.set_text(text)

    return labels


def rgb_to_float(l):
    return np.array(l) / 255.
