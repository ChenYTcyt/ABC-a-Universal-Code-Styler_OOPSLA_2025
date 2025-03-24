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


def lib_edit_dis(hyps, refs, weight_dict):
    if len(refs) == 0 or len(hyps) == 0:
        return None
    score = np.zeros((len(hyps), len(refs)))
    for i in range(len(hyps)):
        for j in range(len(refs)):
            score[i][j] = 1 - sed.sed(hyps[i], refs[j]) / max(max(len(hyps[i]), len(refs[j])), 1)
    score = np.max(score, axis=1)
    weights = []
    for i in range(len(hyps)):
        try:
            weights.append(weight_dict[hyps[i]])
        except KeyError:
            weights.append(0)
    weights = np.array(weights) / np.linalg.norm(weights, ord=1)
    return np.average(score, weights=weights)


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


def extract_lib(codes, parser, lang):
    api_seq = []
    for i in range(len(codes)):
        tmp = []
        tree = parser.parse(bytes(codes[i], 'utf8'))
        if lang == 'cpp':
            _get_lib_names_cpp(codes[i], tree.root_node, tmp)
        elif lang == 'python':
            _get_api_seq_python(codes[i], tree.root_node, tmp)
        elif lang == 'java':
            _get_api_seq_java(codes[i], tree.root_node, tmp)
        tmp = set(tmp)
        api_seq.append(list(tmp))
    return api_seq


def _get_lib_names_cpp(code, node, lib_seq, tmp=None):
    if tmp is None:
        tmp = []
    
    if len(node.children) > 0:
        for child in node.children:
            if node.type == "preproc_include":
                for subchild in node.children:
                    if subchild.type == "string_literal" or subchild.type == "system_lib_string":
                        tmp.append(span_select(subchild, code=code))
            _get_lib_names_cpp(code, child, lib_seq, tmp)
    elif node.type in ["string_literal", "system_lib_string"]:
        if tmp is not None:
            tmp.append(span_select(node, code=code))
    
    if node.type == "translation_unit" and tmp:
        lib_seq.extend(sorted(set(tmp)))


def _get_api_seq_python(code, node, api_seq, tmp=None):
    if node.type == "call":
        api = node.child_by_field_name("function")
        if tmp:
            tmp.append(span_select(api, code=code))
            ant = False
        else:
            tmp = [span_select(api, code=code)]
            ant = True
        for child in node.children:
            _get_api_seq_python(code, child, api_seq, tmp)
        if ant:
            api_seq += tmp[::-1]
            tmp = None
    else:
        for child in node.children:
            _get_api_seq_python(code, child, api_seq, tmp)


def _get_api_seq_java(code, node, api_seq):
    if node.type == "method_invocation":
        obj = node.child_by_field_name("object")
        func = node.child_by_field_name("name")
        if obj:
            api_seq.append(span_select(obj, code=code) + "." + span_select(func, code=code))
        else:
            api_seq.append(span_select(func, code=code))
    else:
        for child in node.children:
            _get_api_seq_java(code, child, api_seq)


def span_select(*nodes, code, indent=False):
    if not nodes:
        return ""
    start, end = nodes[0].start_byte, nodes[-1].end_byte
    select = code[start:end]
    if indent:
        return " " * nodes[0].start_point[1] + select
    return select


def api_postprocess(api_list):
    return list(set(api_list))


def var_postprocess(var_list):
    return list(set(var_list))


def cal_codestyle_dis(hyp, ref, lang, weight_api):
    CPP_LANGUAGE = Language('./metrics_utils/build/my-languages.so', 'cpp')
    parser = Parser()
    parser.set_language(CPP_LANGUAGE)

    hyp_api = extract_lib([hyp], parser, lang)[0]
    hyp_api = api_postprocess(hyp_api)
    ref_api = extract_lib([ref], parser, lang)[0]
    ref_api = api_postprocess(ref_api)
    if len(hyp_api) == 0 or len(ref_api) == 0:
        api_dis = None
    else:
        api_dis = min(lib_edit_dis(hyp_api, ref_api, weight_api), lib_edit_dis(ref_api, hyp_api, weight_api))

    return api_dis


def cal_idf(hyps, refs, phase, lang):
    count = {}
    for code in hyps:
        if phase == 'api':
            apis = extract_lib([code], parser, lang)[0]
            items = api_postprocess(apis)
        for item in items:
            if item in count:
                count[item] += 1
            else:
                count[item] = 1
    for code in refs:
        if phase == 'api':
            apis = extract_lib([code], parser, lang)[0]
            items = api_postprocess(apis)
        for item in items:
            if item in count:
                count[item] += 1
            else:
                count[item] = 1
    total = 0
    for value in count.values():
        total += value
    for key in count.keys():
        if total != count[key]:
            count[key] = math.log(total / count[key], math.e)
        else:
            count[key] = 1
    return count


def get_overall_csd(code, aug, lang, weight_api):
    api_dis = cal_codestyle_dis(code, aug, lang, weight_api)
    return {
        'Library': api_dis if api_dis is not None else 0
    }


def compute_library_by_list(refs, preds):
    CPP_LANGUAGE = Language('./metrics_utils/build/my-languages.so', 'cpp')
    parser = Parser()
    parser.set_language(CPP_LANGUAGE)

    api_idf = cal_idf(preds, refs, 'api', 'cpp')

    length = len(refs)
    score_average = { 'Library': 0 }
    for i in range(0, length):
        score = get_overall_csd(preds[i], refs[i], 'cpp', api_idf)
        for key in score_average:
            score_average[key] = score_average[key] + score.get(key, 0)
    for key in score_average:
        score_average[key] = round(score_average[key]/length*100, 3)
    return score_average