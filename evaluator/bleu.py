# -*- coding: utf-8 -*-

"""
@Author             : Bao
@Date               : 2020/1/1 00:00
@Desc               :
@Last modified by   : Bao
@Last modified date : 2020/4/5 19:37
"""

from nltk.translate.bleu_score import corpus_bleu

from utils import read_json_lines


class Evaluator:
    def __init__(self, key):
        self.key = key

    def evaluate(self, ref_file, hyp_file, to_lower):
        list_of_references = []
        for line in read_json_lines(ref_file):
            ref = line[self.key]  # ref is a list of words
            if to_lower:
                ref = list(map(str.lower, ref))
            list_of_references.append([ref])

        hypotheses = []
        for line in read_json_lines(hyp_file):
            hyp = line[self.key]  # hyp is a list of words
            if to_lower:
                hyp = list(map(str.lower, hyp))
            hypotheses.append(hyp)

        assert len(list_of_references) == len(hypotheses)

        bleu1 = corpus_bleu(list_of_references, hypotheses, (1., 0., 0., 0.))
        bleu2 = corpus_bleu(list_of_references, hypotheses, (0.5, 0.5, 0., 0.))
        bleu3 = corpus_bleu(list_of_references, hypotheses, (0.33, 0.33, 0.33, 0.))
        bleu4 = corpus_bleu(list_of_references, hypotheses, (0.25, 0.25, 0.25, 0.25))
        result = {
            'Bleu_1': bleu1,
            'Bleu_2': bleu2,
            'Bleu_3': bleu3,
            'Bleu_4': bleu4,
        }
        for k, v in result.items():
            print('{}: {}'.format(k, v))
        return result
