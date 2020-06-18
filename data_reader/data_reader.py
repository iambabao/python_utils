# -*- coding: utf-8 -*-

"""
@Author             : Bao
@Date               : 2020/1/1 00:00
@Desc               :
@Last modified by   : Bao
@Last modified date : 2020/6/19 19:37
"""

import json
from tqdm import tqdm

from utils import read_json_lines, convert_list


class DataReader:
    def __init__(self, config):
        self.config = config

    def _read_data(self, data_file):
        src_seq = []
        tgt_seq = []

        counter = 0
        data_iter = tqdm(list(read_json_lines(data_file)))
        for line in data_iter:
            line = json.loads(line)
            src = line.get('src', [''])
            tgt = line.get('tgt', [''])

            if self.config.to_lower:
                src = list(map(str.lower, src))
                tgt = list(map(str.lower, tgt))

            src = src[:self.config.sequence_len]
            tgt = [self.config.sos] + tgt[:self.config.sequence_len-2] + [self.config.eos]

            src_seq.append(convert_list(src, self.config.src_2_id, self.config.pad_id, self.config.unk_id))
            tgt_seq.append(convert_list(tgt, self.config.tgt_2_id, self.config.pad_id, self.config.unk_id))

        return src_seq, tgt_seq

    def read_train_data(self):
        return self._read_data(self.config.train_data)

    def read_valid_data(self):
        return self._read_data(self.config.valid_data)

    def read_test_data(self):
        return self._read_data(self.config.test_data)
