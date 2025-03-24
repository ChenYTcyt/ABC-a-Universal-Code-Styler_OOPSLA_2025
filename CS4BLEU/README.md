

# Evaluation Metric

## CS4BLEU

The calculation of CS4BLEU (Code Sequence, Syntactic, Semantic and Stylistic BLEU) is shown in `cs4bleu.py`.


## BLEU
The function `compute_bleu_by_tokens_list` in `./metrics_utils/bleu.py` calculates BLEU between two code snippets.

## Dataflow
The function `compute_dataflow_match` in `./metrics_utils/dataflow_match.py` calculates dataflow matching between two code snippets. 

## AST
The function `compute_ast_by_list` in `./metrics_utils/ast.py` calculates code structure similarity between two code snippets.

## Library
The function `compute_library_by_list` in `./metrics_utils/library.py` calculates library calls similarity between two code snippets.
