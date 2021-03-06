# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Processing data for pretraining."""

import argparse
import json
import multiprocessing
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import time

import torch
try:
    import nltk
    nltk_available = True
except ImportError:
    nltk_available = False

from data_utils.tokenization_gpt2 import GPT2Tokenizer
from data import indexed_dataset


# https://stackoverflow.com/questions/33139531/preserve-empty-lines-with-nltks-punkt-tokenizer
class CustomLanguageVars(nltk.tokenize.punkt.PunktLanguageVars):

    _period_context_fmt = r"""
        \S*                          # some word material
        %(SentEndChars)s             # a potential sentence ending
        \s*                       #  <-- THIS is what I changed
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
        ))"""

class IdentitySplitter(object):
    def tokenize(self, *text):
        return text

class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = GPT2Tokenizer(os.path.join(self.args.tokenizer_path, 'vocab.txt'))

        Encoder.splitter = IdentitySplitter()

    #def encode(self, json_line):
    #    data = json.loads(json_line)
    #    ids = {}
    #    for key in self.args.json_keys:
    #        text = data[key]
    #        doc_ids = []
    #        for sentence in Encoder.splitter.tokenize(text):
    #            sentence_ids = Encoder.tokenizer.tokenize(sentence)
    #            if len(sentence_ids) > 0:
    #                doc_ids.append(sentence_ids)
    #        if len(doc_ids) == 0:
    #            doc_ids.append([])
    #        if self.args.append_eod:
    #            doc_ids[-1].append(Encoder.tokenizer.eod)
    #        ids[key] = doc_ids
    #    return ids, len(json_line)

    def encode(self, line):
        data = line.strip()
        ids = {}
        doc_ids = Encoder.tokenizer.encode(data)
        doc_ids.append(Encoder.tokenizer.eod)
        ids['text'] = [doc_ids]
        return ids, len(line)


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help='Path to input JSON')
    group.add_argument('--json-keys', nargs='+', default=['text'],
                       help='space separate listed of keys to extract from json')
    #group.add_argument('--split-sentences', action='store_true',
    #                   help='Split documents into sentences.')
    #group.add_argument('--keep-newlines', action='store_true',
    #                   help='Keep newlines between sentences when splitting.')

    group = parser.add_argument_group(title='tokenizer')
    #group.add_argument('--tokenizer-type', type=str, required=True,
    #                   choices=['BertWordPieceLowerCase','BertWordPieceCase',
    #                            'GPT2BPETokenizer'],
    #                   help='What type of tokenizer to use.')
    #group.add_argument('--vocab-file', type=str, default=None,
    #                   help='Path to the vocab file')
    #group.add_argument('--merge-file', type=str, default=None,
    #                   help='Path to the BPE merge file (if necessary).')
    #group.add_argument('--append-eod', action='store_true',
    #                   help='Append an <eod> token to the end of a document.')
    group.add_argument('--tokenizer-path', type=str, help='Path of tokenizer')

    group = parser.add_argument_group(title='output data')
    group.add_argument("--output-path", type=str, required=True)
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')
    group.add_argument('--dataset-impl', type=str, default='mmap',
                       choices=['lazy', 'cached', 'mmap'])

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, default=1,
                       help='Number of worker processes to launch')
    group.add_argument('--log-interval', type=int, default=10000,
                       help='Interval between progress updates')
    args = parser.parse_args()
    args.keep_empty = False

    #if args.tokenizer_type.lower().startswith('bert'):
    #    if not args.split_sentences:
    #        print("Bert tokenizer detected, are you sure you don't want to split sentences?")

    # some default/dummy values for the tokenizer
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    #args.model_parallel_size = 1

    return args

def main():
    args = get_args()
    startup_start = time.time()

    print("Opening", args.input)
    fin = open(args.input, 'r', encoding='utf-8')

    encoder = Encoder(args)
    tokenizer = GPT2Tokenizer(os.path.join(args.tokenizer_path, 'vocab.txt'))
    pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)
    encoded_docs = pool.imap(encoder.encode, fin, 25)
    #encoded_docs = map(encoder.encode, fin)

    level = "document"

    #print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Output prefix: {args.output_prefix}")
    output_bin_files = {}
    output_idx_files = {}
    builders = {}
    for key in args.json_keys:
        output_bin_files[key] = os.path.join(args.output_path, "{}_{}_{}.bin".format(args.output_prefix, key, level))
        output_idx_files[key] = os.path.join(args.output_path,  "{}_{}_{}.idx".format(args.output_prefix, key, level))
        builders[key] = indexed_dataset.make_builder(output_bin_files[key],
                                               impl=args.dataset_impl,
                                               vocab_size=tokenizer.vocab_size)

    startup_end = time.time()
    proc_start = time.time()
    total_bytes_processed = 0
    print("Time to startup:", startup_end - startup_start)

    for i, (doc, bytes_processed) in enumerate(encoded_docs, start=1):
        total_bytes_processed += bytes_processed
        for key, sentences in doc.items():
            for sentence in sentences:
                builders[key].add_item(torch.IntTensor(sentence))
            builders[key].end_document()
        if i % args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed / elapsed / 1024 / 1024
            print(f"Processed {i} documents",
                  f"({i/elapsed} docs/s, {mbs} MB/s).",
                  file=sys.stderr)

    for key in args.json_keys:
        builders[key].finalize(output_idx_files[key])

if __name__ == '__main__':
    main()
