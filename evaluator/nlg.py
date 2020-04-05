# -*- coding: utf-8 -*-

"""
@Author             : Bao
@Date               : 2020/1/1 00:00
@Desc               :
@Last modified by   : Bao
@Last modified date : 2020/4/5 19:37
"""

from .pycocoevalcap.bleu.bleu import Bleu
from .pycocoevalcap.meteor.meteor import Meteor
from .pycocoevalcap.rouge.rouge import Rouge
from .pycocoevalcap.cider.cider import Cider
from utils import read_json_lines


class Evaluator:
    def __init__(self, key):
        self.key = key

    def evaluate(self, ref_file, hyp_file, to_lower):
        refs = []
        for line in read_json_lines(ref_file):
            ref = line[self.key]  # ref is a sentence
            if to_lower:
                ref = ref.lower()
            refs.append(ref)
        refs = {idx: [ref] for idx, ref in enumerate(refs)}

        hyps = []
        for line in read_json_lines(hyp_file):
            hyp = line[self.key]  # hyp is a sentence
            if to_lower:
                hyp = hyp.lower()
            hyps.append(hyp)
        hyps = {idx: [hyp] for idx, hyp in enumerate(hyps)}

        assert len(refs) == len(hyps)

        res = {}
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]
        for scorer, method in scorers:
            score, scores = scorer.compute_score(refs, hyps)
            if isinstance(method, list):
                for sc, scs, m in zip(score, scores, method):
                    print("%s: %0.6f" % (m, sc))
                    res[m] = sc
            else:
                print("%s: %0.6f" % (method, score))
                res[method] = score

        return res
