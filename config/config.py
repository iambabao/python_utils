# -*- coding: utf-8 -*-

"""
@Author             : Bao
@Date               : 2020/1/1 00:00
@Desc               :
@Last modified by   : Bao
@Last modified date : 2020/6/18 19:37
"""

import os


class Config:
    def __init__(self, root_dir, current_model,
                 num_epoch=30, batch_size=32, top_k=5, threshold=0.4,
                 max_num_seqs=20, max_seq_length=128,
                 word_em_size=300, feature_em_size = 50,
                 hidden_size=256, attention_size=256,
                 kernel_size=(2, 3, 4, 5), filter_dim=64,
                 num_layers=4, num_heads=8, model_dim=256,
                 fc_size_s=128, fc_size_m=512, fc_size_l=1024,
                 optimizer='Adam', lr=0.001, dropout=0.1, l2_rate=0.0,
                 beam_search=False):
        self.root_dir = root_dir

        self.temp_dir = os.path.join(self.root_dir, 'temp')

        self.data_dir = os.path.join(self.root_dir, 'data')
        self.train_data = os.path.join(self.data_dir, 'data_train.json')
        self.valid_data = os.path.join(self.data_dir, 'data_valid.json')
        self.test_data = os.path.join(self.data_dir, 'data_test.json')
        self.stop_words = os.path.join(self.data_dir, 'stop_words.txt')
        self.vocab_dict = os.path.join(self.data_dir, 'dict_vocab.json')
        self.src_vocab_dict = os.path.join(self.data_dir, 'dict_src_vocab.json')
        self.tgt_vocab_dict = os.path.join(self.data_dir, 'dict_tgt_vocab.json')
        self.label_dict = os.path.join(self.data_dir, 'dict_label.json')

        self.embedding_dir = os.path.join(self.data_dir, 'embedding')
        self.plain_text = os.path.join(self.embedding_dir, 'plain_text.txt')
        self.word2vec_model = os.path.join(self.embedding_dir, 'word2vec.model')
        self.glove_file = os.path.join(self.embedding_dir, 'glove.6B.300d.txt')

        self.current_model = current_model
        self.result_dir = os.path.join(self.root_dir, 'results')
        self.model_file = os.path.join(self.result_dir, self.current_model, 'model')
        self.valid_outputs = os.path.join(self.result_dir, self.current_model, 'valid_outputs.json')
        self.valid_results = os.path.join(self.result_dir, self.current_model, 'valid_results.json')
        self.test_outputs = os.path.join(self.result_dir, self.current_model, 'test_outputs.json')
        self.test_results = os.path.join(self.result_dir, self.current_model, 'test_results.json')
        self.train_log_dir = os.path.join(self.result_dir, self.current_model, 'train_log')
        self.valid_log_dir = os.path.join(self.result_dir, self.current_model, 'valid_log')

        # BERT
        self.bert_dir = os.path.join(self.root_dir, 'chinese_L-12_H-768_A-12')
        self.bert_vocab = os.path.join(self.bert_dir, 'vocab.txt')
        self.bert_config = os.path.join(self.bert_dir, 'bert_config.json')
        self.bert_ckpt = os.path.join(self.bert_dir, 'bert_model.ckpt')

        self.pad = '<pad>'
        self.pad_id = 0
        self.unk = '<unk>'
        self.unk_id = 1
        self.sos = '<sos>'
        self.sos_id = 2
        self.eos = '<eos>'
        self.eos_id = 3
        self.sep = '<sep>'
        self.sep_id = 4
        self.num = '<num>'
        self.num_id = 5
        self.time = '<time>'
        self.time_id = 6
        self.vocab_size = 80000
        self.src_vocab_size = 40000
        self.tgt_vocab_size = 40000
        self.word_2_id = {}
        self.id_2_word = {}
        self.to_lower = True

        self.top_k = top_k
        self.threshold = threshold
        self.max_num_seqs = max_num_seqs
        self.max_seq_length = max_seq_length
        self.word_em_size = word_em_size
        self.feature_em_size = feature_em_size
        self.beam_search = beam_search

        # RNN
        self.hidden_size = hidden_size
        self.attention_size = attention_size

        # CNN
        self.kernel_size = kernel_size
        self.filter_dim = filter_dim

        # Transformer
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.model_dim = model_dim

        # FC
        self.fc_size_s = fc_size_s
        self.fc_size_m = fc_size_m
        self.fc_size_l = fc_size_l

        # Train
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.lr = lr
        self.dropout = dropout
        self.l2_rate = l2_rate
