import re
import json

from metrics_utils.bleu import compute_bleu_by_tokens_list
from metrics_utils.dataflow_match import compute_dataflow_match
from metrics_utils.ast import compute_ast_by_list
from metrics_utils.library import compute_library_by_list

def tokenize_cpp_code(code):
    token_pattern = r'''
        std::
        | bits/stdc\+\+.h
        | [a-zA-Z_]\w*
        | \d+\.\d+|\d+
        | [{}()\[\];:,<>.]
        | ->|::|&&|\|\||==|!=|<=|>=|<<|>>
        | [+\-*/%&|^~!]
    '''
    tokens = re.findall(token_pattern, code, re.VERBOSE)
    return tokens


def evaluate_cs4bleu(pred_file):
    pred_data = json.load(open(pred_file, 'r'))
    refs = [data['ref'] for data in pred_data]
    preds = [data['real_pred'] for data in pred_data]
    tokenized_refs = [data['tokenized_ref'] for data in pred_data]
    tokenized_preds = [data['tokenized_pred'] for data in pred_data]
    assert len(refs) == len(preds)
    evaluation = {}
    evaluation.update(compute_bleu_by_tokens_list(tokenized_refs, tokenized_preds))
    evaluation.update(compute_dataflow_match(tokenized_refs, tokenized_preds, refs, preds))
    evaluation.update(compute_ast_by_list(refs, preds))
    evaluation.update(compute_library_by_list(refs, preds))
    cs4bleu = (evaluation['CSSim']['Lib'] + evaluation['CSSim']['Tree'] + evaluation['BLEU']['BLEU'] + evaluation['CodeBLEU']['Dataflow Match']) / 4
    cs4bleu = (
        evaluation['BLEU']['BLEU']
        + evaluation['Dataflow Match']
        + evaluation['AST']
        + evaluation['Library']
    )
    evaluation.update({
        'CS4BLEU': round(cs4bleu, 3)
    })
    return evaluation