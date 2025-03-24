import math
import edist.sed as sed
import edist.ted as ted
import numpy as np
import json
from tqdm import trange
from tree_sitter import Language, Parser
import warnings
warnings.simplefilter('ignore', FutureWarning)
CPP_LANGUAGE = Language('./metrics_utils/build/my-languages.so', 'cpp')
parser = Parser()
parser.set_language(CPP_LANGUAGE)


def tree_edit_dis(hyp, ref, parser):
    tree = parser.parse(bytes(hyp, 'utf8')).root_node
    all_nodes1 = []
    get_all_subtrees(tree, all_nodes1)
    hyp_nodes, hyp_adj = extract_subtrees(all_nodes1[0])
    tree = parser.parse(bytes(ref, 'utf8')).root_node
    all_nodes2 = []
    get_all_subtrees(tree, all_nodes2)
    ref_nodes, ref_adj = extract_subtrees(all_nodes2[0])
    tree_dis = ted.standard_ted(hyp_nodes, hyp_adj, ref_nodes, ref_adj)

    return tree_dis / max(len(hyp_nodes), len(ref_nodes))


def extract_subtrees(tree):
    nodes, adj_nodes = [], []
    name_dict, num_dict = {}, {}
    traverse_tree(tree, nodes, adj_nodes, name_dict, num_dict)

    nodes = [name_dict[node] for node in nodes]
    children_index = []
    for children in adj_nodes:
        tmp = []
        for node in children:
            tmp.append(num_dict[node])
        children_index.append(tmp)
    return nodes, children_index


def get_all_subtrees(tree, nodes):
    if tree.children is not None:
        nodes.append(tree)

    for child in tree.children:
        get_all_subtrees(child, nodes)


def traverse_tree(tree, nodes, adj_nodes, name_dict=None, num_dict=None):
    if tree.children is not None:
        nodes.append(tree.id)
        adj_nodes.append([node.id for node in tree.children])
        name_dict[tree.id] = tree.type
        num_dict[tree.id] = len(num_dict)

    for child in tree.children:
        traverse_tree(child, nodes, adj_nodes, name_dict, num_dict)


def span_select(*nodes, code, indent=False):
    if not nodes:
        return ""
    start, end = nodes[0].start_byte, nodes[-1].end_byte
    select = code[start:end]
    if indent:
        return " " * nodes[0].start_point[1] + select
    return select


def cal_ast_dis(hyp, ref):
    CPP_LANGUAGE = Language('./metrics_utils/build/my-languages.so', 'cpp')
    parser = Parser()
    parser.set_language(CPP_LANGUAGE)

    tree_edit_distance = tree_edit_dis(hyp, ref, parser)
    tree_dis = 1 - tree_edit_distance

    return tree_dis


def get_overall_ast(code, aug, lang):
    tree_dis = cal_ast_dis(code, aug, lang)
    return {
        'AST': tree_dis
    }


def compute_ast_by_list(refs, preds):
    CPP_LANGUAGE = Language('./metrics_utils/build/my-languages.so', 'cpp')
    parser = Parser()
    parser.set_language(CPP_LANGUAGE)

    length = len(refs)
    score_average = { 'AST': 0 }
    for i in range(0, length):
        score = get_overall_ast(preds[i], refs[i], 'cpp')
        for key in score_average:
            score_average[key] = score_average[key] + score.get(key, 0)
    for key in score_average:
        score_average[key] = round(score_average[key]/length*100, 3)
    return score_average