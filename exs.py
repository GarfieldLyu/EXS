from typing import List, Dict, Callable, Tuple, Union
import numpy as np
from sklearn.utils import check_random_state
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression, SGDClassifier
import matplotlib.pyplot as plt
from tqdm import tqdm
import nltk


class ExplainableSearch:
    # Implementation for paper EXS: Explainable Search Using Local Model Agnostic Interpretability.
    # link: https://arxiv.org/pdf/1809.03857.pdf

    def __init__(self, ranker: Callable, exs_model: str, num_samples: int=1000, seed: int=10) -> None:
        """
        Explain a black-box ranker model.
        Args:
            ranker: a ranking model, ranker.predict([query_text, doc_text]) = score;
            
            exs_model: a simpler machine learning model.
            num_samples: the number of total samples created for learning exs_model. 1000 by default.
        """
        self.num_samples = num_samples
        self.batch_size = 10
        self.ranker = ranker
        self.exs_model = exs_model
        self.random_state = check_random_state(seed)
    
    def _set_exs_model(self, exs_model: str, seed: int=10) -> None:
        """ Update the surrogate model."""
        if exs_model == 'lr':
            return LogisticRegression(random_state=seed)
        elif exs_model == 'svm':
            return SGDClassifier(random_state=seed)
        else:
            raise NotImplementedError('Only sopport . :(')

    def explain(self, corpus: Dict[str, Dict[str, float]], docs_exp: Dict[str, str], topk: Union[int, Dict[str, int], List[str]], Method: Union[str, Dict[str, str]], seed: int=10) -> Dict[str, np.array]:
        """ Explain the rank for a group of queries. 
            Args:
                corpus: the query-doc-rerank datasets. The format should follow:
                    {'query1': {'doc1': rel_score, 'doc2': rel_score2,...}, 'query2': {...}, ...}
                docs_exp: the doc to be explained. {query: which_doc}, this doc has to be the raw text.
                topk: integer, the baseline doc's rank which is used to explain the doc in doc_ids
        
        """
        Results = {}
        for query in corpus:
            if isinstance(topk, dict):  # choose different baseline doc for each query.
                k = topk[query]
            else:
                k = topk
            if isinstance(Method, dict):  # choose different method for each query.
                method = Method[query]
            else:
                method = Method
            doc_exp = docs_exp[query]   # the doc to be explained.
            docs_sorted = sorted(corpus[query].items(), key=lambda item: item[1], reverse=True)[:k]
            scores_topk = [d[1] for d in docs_sorted] 
            results_exp = self.explain_single(query, doc_exp, scores_topk, method, seed)
            Results[query] = results_exp
        return Results

    def explain_single(self, query: str, doc_h: str, scores_topk: List[float], method: str, seed) -> Tuple[np.array, np.array]:
        """ explain a single doc for a single query.
            Args: 
                query: raw-text query. NOT query_id!
                doc_h: the doc to be explained, it should have higher rank than k. raw text!
                scores_topk: the list of topk ranking scores by ranker. In descending order.
                method: three methods for generating labels for perturbed docs to train the surrogate model.

         """
        docs_perturb = self.perturb_doc(doc_h)
        lables_perturb = self.generate_label(query, docs_perturb, scores_topk, method)
        """ Learn a simpler model"""
        
        clf = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('clf', self._set_exs_model(self.exs_model, seed))])

        clf.fit(docs_perturb, lables_perturb)
        coef = clf['clf'].coef_.copy()
        vocabs = np.array(clf['vect'].get_feature_names())
        return vocabs, coef

    def perturb_doc(self, doc_h: str) -> np.array:
        """ Given a text doc_h, generate a list of perturbed doc, by randomly removing words in doc_h."""
        tokens = nltk.word_tokenize(doc_h)
        tokens_len = len(tokens)
        sample = self.random_state.randint(1, tokens_len + 1, self.num_samples - 1)
        
        all_docs = [doc_h]
        for _, size in tqdm(enumerate(sample)):
            remove_pos = self.random_state.choice(range(tokens_len), size, replace=False)
            doc_adv = ' '.join([tok for i, tok in enumerate(tokens) if i not in remove_pos])
            all_docs.append(doc_adv)
        all_docs = np.array(all_docs)
        return all_docs

    def generate_label(self, query: str, all_docs: List[str], scores_topk: List[float], method: str='topk-bin') -> np.array:
        """ Generate labels for all perturbed docs, based on three methods described in the paper.
            Args:
                query: raw-text query.
                all_docs: a list of perturbed docs based on doc_exp.
                scores_topk: the topk ranking scores in descending order by the ranker.
        """
        input_pairs = [[query, doc] for doc in all_docs]
        rank_scores = self.ranker.predict(input_pairs, batch_size=self.batch_size)
        score_l = scores_topk[-1]
        score_h = scores_topk[0]

        """ According to the paper, score_based and rank_based generate a float instead of binary labels."""
        if method == 'topk-bin':
            labels = (rank_scores > score_l).astype(int)
        elif method == 'score':
            labels =  (rank_scores > score_h).astype(int)
        elif method == 'rank':
            #rank_new = np.argsort(scores_topk.append())
            #labels = [0 if score < score_l else 1 - rank_new/len(scores_topk) for score in rank_scores]
            pass
        else:
            raise ValueError('Invalid method.')
        return labels


    def visualize(self, vocabs: np.array, coef: np.array, show_top: int=10, saveto: str='exs.pdf'):
        if len(coef.shape) > 1:  # binary, 
            coef = np.squeeze(coef)
        sorted_coef = np.sort(coef)
        sorted_idx = np.argsort(coef)
        pos_y = sorted_coef[-show_top:]
        neg_y = sorted_coef[:show_top]
        pos_idx = sorted_idx[-show_top:]
        neg_idx = sorted_idx[:show_top]

        words = np.append(vocabs[pos_idx], vocabs[neg_idx])
        y = np.append(pos_y, neg_y)

        fig, ax = plt.subplots(figsize=(8, 10))
        colors = ['green' if val >0 else 'red' for val in y]
        pos = np.arange(len(y)) + .5
        ax.barh(pos, y, align='center', color=colors)
        ax.set_yticklabels(words, fontsize=18)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        fig.savefig(saveto)
