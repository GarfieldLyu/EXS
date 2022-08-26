from typing import Any, Callable, Dict, Tuple, List, Union
from pyserini.index.lucene import IndexReader
import math
import numpy as np
from itertools import combinations
import random
from functools import partial

# check the manual for building index at: https://github.com/castorini/pyserini#how-do-i-index-and-search-my-own-documents
index_path = 'datasets/dbpedia-entity/pyserini/dbpedia-entity-small.index'
params = {'top_idf': 10, 'topk': 5, 'max_pair': 100, 'max_intent': 10, 'style':'random'}



class IntentEXS:
    # Implementation for "Model Agnostic Interpretability of Rankers via Intent Modelling".
    # https://dl.acm.org/doi/pdf/10.1145/3351095.3375234
    def __init__(self, ranker: Callable, index_path: str, exp_model: str, seed: int=10) -> None:
        """ Init the ranking model to be explained, a pre-computed index, a simple explaination model, only support BM25."""
        self.ranker = ranker
        self.indexer = IndexReader(index_path)  # init a pre-computed index.
      
        if exp_model.lower() == 'bm25':
            self.exp_model = self._bm25_model
        else:
            raise NotImplementedError(f'Only support bm25.')

        self.seed = seed
        self. _gen_candidates = partial(gen_candidates, self.indexer, self.ranker)
        self._gen_pairs = partial(gen_pairs, seed=self.seed)
        self._gen_matrix = partial(gen_matrix, self.exp_model)
        self._optimize = greedy

    def _bm25_model(self, doc_id: str, term: str):
        """ Use BM25 model as the explainer."""
        return self.indexer.compute_bm25_term_weight(doc_id, term, analyzer=None)
    
    def explain(self, corpus: Dict[str, Any], params: Dict[str, Union[int, str]]) -> List[str]:
        """ A pipeline including candidates, matrix generation, and extract intent using greedy algorithm.
            Args:
                corpus: dict-type input data, must have query, scores, docs domains. 
                        e.g. {'query': xxx, 'scores': {'doc_id': score}, 'docs': {'doc': doc_text}}.
                params: necessary parameters needed for candidates gen, matrix gen... 
                        e.g. {'top_idf': 10, 'topk': 5, 'max_pair': 100, 'max_intent': 10}
            Return:
                A list of terms/words/tokens, used as expansion/intent/explanation.
        
        """
        doc_ids = [score[0] for score in sorted(corpus['scores'].items(), key=lambda item: item[1], reverse=True) ] # sorted in descending order.
        candidates = self._gen_candidates(corpus, params['top_idf'])
        self.candidates = candidates
        doc_pairs = self._gen_pairs(params['style'], len(doc_ids), params['topk'], params['max_pair'])
        #print(f'sampled pairs: {len(doc_pairs)}')
        matrix = self._gen_matrix(doc_ids, candidates, doc_pairs)
        self.matrix = matrix
        expansion = self._optimize(candidates, matrix, params['max_intent'])
        return expansion






def gen_candidates(indexer: IndexReader, ranker: Callable, corpus: Dict[str, Dict[str, Any]], top_idf: int=10) -> List[str]:
    """ Generates candidate tokens for documents of a query."""
    query = corpus['query']
    doc_ids = list(corpus['scores'].keys())
    candidates = []
    for doc_id in doc_ids:
        #print(doc_id)
        doc = corpus['docs'][doc_id]
        score = corpus['scores'][doc_id]
        terms_sorted = terms_tfidf(indexer, doc_id)[:2*top_idf]   # keep 2*top_idf terms for perturbation for now.
        new_input_pairs = [(query, doc_perturb(doc, term[0])) for term in terms_sorted]
        scores_new = ranker.predict(new_input_pairs)
        term_idx = np.argsort(-np.abs(scores_new - score))[:top_idf]   # descending order.
        terms_select = np.array([term[0] for term in terms_sorted])[term_idx].tolist()
        candidates.extend(terms_select)
    candidates = list(set(candidates))
    return candidates

def gen_pairs(style: str, length: int, topk: int, max_pair: int, seed: int) -> List[Tuple[int, int]]:
    """ Sample document pairs by rank. e.g., [(0, 5), (1, 9), (3, 6),...]"""
    if style == 'random':
        pairs = list(combinations(range(length), 2))
    elif style == 'topk_random':
        assert(topk <= length)
        ranked_list = list(range(topk))
        tail_list = list(range(topk, length))
        pairs = [(a, b) for a in ranked_list for b in tail_list]
    elif style == 'topk_rank_random':
        pairs = list(combinations(range(topk), 2))
    else:
        raise ValueError(f'Not supported style {style}')
    
    if len(pairs) < max_pair:
        max_pair = len(pairs)
    random.seed(seed)    
    pairs = random.sample(pairs, max_pair)
    return pairs

def gen_matrix(exp_model: Callable, doc_ids: List[str], candidates: List[str], pairs: List[Tuple]) -> np.array:
    """ Generate the matrix, given candidates and sampled document pairs."""
    matrix = []
    idx_set = set([p for pair in pairs for p in pair])
    # compute all bm25 scores.
    BM25_scores = {}
    for idx in idx_set:
        doc_id = doc_ids[idx]
        BM25_scores[doc_id] = np.array([exp_model(doc_id, term) for term in candidates])

    for rank_h, rank_l in pairs:
        docid_h, docid_l = doc_ids[rank_h], doc_ids[rank_l]
        bm25_scores_h, bm25_scores_l = BM25_scores[docid_h], BM25_scores[docid_l]
        column = (bm25_scores_h - bm25_scores_l) * (1 + math.log(rank_l - rank_h))
        matrix.append(column)
    
    matrix = np.array(matrix).transpose(1, 0)   # terms at dimension first. 
    return matrix

def greedy(candidates: List[str], matrix_arg: np.array, select_max: int) -> List[str]:
    matrix = matrix_arg.tolist()  # copy the argument, otherwise it'll be modified.     
    #print(matrix.shape)
    expansion = []
    pcov = np.zeros(len(matrix[0]))   # init expansion and features
    for _ in range(select_max):
        covered = (pcov > 0.0).sum()
        pcov_update = pcov.copy()
        picked = None   # in case no candidate can be found.
        for candidate, array in zip(candidates, matrix):
            pcov_expand = pcov + array   
            u = (pcov_expand > 0.0).sum()
            if covered < u:   # always pick the one with largest utility.
                covered = u
                picked = candidate
                pcov_update = pcov_expand.copy()

        if picked:
            print('Picked!')
            expansion.append(picked)
            pcov = pcov_update.copy()
            picked_id = candidates.index(picked)
            del candidates[picked_id]  # remove the picked item from candidates.
            del matrix[picked_id]  # remove the picked features from matrix
        else:  # greedy select algorithm stops here, because the utility does not improve anymore.
            break
    return expansion


def terms_tfidf(indexer: IndexReader, doc_id: str) -> List[Tuple[str, float]]:
    """ Given a document id, return the terms sorted by tf-idf scores.
    Args:
        doc_id: str, the id of doc in corpus.

    Return:
        terms with tf-idf score, sorted, a list of tuple. 
    """
    num_docs = indexer.stats()['documents']  # the number of documents
    tf = indexer.get_document_vector(doc_id)
    df = {term: (indexer.get_term_counts(term))[0] for term in tf.keys()}
    tf_idf = {term: tf[term] * math.log(num_docs/(df[term] + 1)) for term in tf.keys() } 
    terms_sorted = sorted(tf_idf.items(), key=lambda x: x[1], reverse=True)
    return terms_sorted

def doc_perturb(doc, term):
    doc_new = doc.replace(f' {term} ', '')
    return doc_new









