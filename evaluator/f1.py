import json
from utils import read_dict


class Evaluator:
    def __init__(self, label_dict):
        self.label_2_id, self.id_2_label = read_dict(label_dict)
        self.label_num = len(self.label_2_id)

    # Gen new results according to the truth and users output
    def gen_new_result(self, result, truth, pred):
        s1 = set()
        for v in pred['label']:
            s1.add(self.label_2_id[v])
        s2 = set()
        for v in truth['label']:
            s2.add(self.label_2_id[v])

        for v in range(self.label_num):
            in1 = v in s1
            in2 = v in s2
            if in1:
                if in2:
                    result[0][v]['TP'] += 1
                else:
                    result[0][v]['FP'] += 1
            else:
                if in2:
                    result[0][v]['FN'] += 1
                else:
                    result[0][v]['TN'] += 1

        return result

    # Calculate precision, recall and f1 value
    # According to https://github.com/dice-group/gerbil/wiki/Precision,-Recall-and-F1-measure
    @staticmethod
    def get_prf(res):
        if res['TP'] == 0:
            if res['FP'] == 0 and res['FN'] == 0:
                precision = 1.0
                recall = 1.0
                f1 = 1.0
            else:
                precision = 0.0
                recall = 0.0
                f1 = 0.0
        else:
            precision = 1.0 * res['TP'] / (res['TP'] + res['FP'])
            recall = 1.0 * res['TP'] / (res['TP'] + res['FN'])
            f1 = 2 * precision * recall / (precision + recall)

        return precision, recall, f1

    def get_result(self, truth_file, pred_file):
        result = [[]]
        for v in range(self.label_num):
            result[0].append({'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0})

        f_truth = open(truth_file, 'r', encoding='utf-8')
        f_pred = open(pred_file, 'r', encoding='utf-8')

        for line in f_truth:
            truth = json.loads(line)
            pred = json.loads(f_pred.readline())

            result = self.gen_new_result(result, truth, pred)

        return result

    def calc_f1(self, result):
        sum_f = 0
        temp = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
        for res in result:
            p, r, f = self.get_prf(res)
            sum_f += f
            for v in res.keys():
                temp[v] += res[v]

        _, _, micro_f = self.get_prf(temp)
        macro_f = sum_f / len(result)

        return micro_f, macro_f
