# -*- coding: utf-8 -*-

"""
@Author             : Bao
@Date               : 2020/1/1 00:00
@Desc               :
@Last modified by   : Bao
@Last modified date : 2020/4/5 19:37
"""

import json

from utils import convert_list


class DataReader:
    def __init__(self, config):
        self.config = config

    def _read_data(self, data_file):
        src_seq = []
        tgt_seq = []

        counter = 0
        with open(data_file, 'r', encoding='utf-8') as fin:
            for line in fin:
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

                counter += 1
                if counter % 10000 == 0:
                    print('\rprocessing file {}: {:>6d}'.format(data_file, counter), end='')
            print()

        return src_seq, tgt_seq

    def read_train_data(self):
        return self._read_data(self.config.train_data)

    def read_valid_data(self):
        return self._read_data(self.config.valid_data)

    def read_test_data(self):
        return self._read_data(self.config.test_data)
