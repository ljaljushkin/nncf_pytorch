"""
 Copyright (c) 2020 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import sys
from argparse import ArgumentParser
from os import replace

import torch
from os import listdir, makedirs
from os.path import isfile, join, exists

from pathlib import Path


def main(argv):
    parser = ArgumentParser()
    parser.add_argument('-i', required=True)
    args = parser.parse_args(args=argv)

    sd = torch.load(args.i)
    if 'state_dict' in sd:
        sd = sd['state_dict']

    new_sd = {}
    for k, v in sd.items():
        old_k = k
        if 'activation_quantizers' in k and 'INPUT' not in k:
            key_split = k.split('.')
            key_split[-2] += '|OUTPUT'
            k = '.'.join(key_split)
            # k = k.replace('ConvBNReLU', 'ConvBNActivation')
            k = k.replace('activation_quantizers', 'external_quantizers')
            print(f'{old_k} -> {k}')
        new_sd[k] = v
    torch.save(new_sd, Path(args.i).parent / 'model_patched.sd')


if __name__ == '__main__':
    main(sys.argv[1:])
