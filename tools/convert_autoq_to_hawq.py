"""
 Copyright (c) 2019 Intel Corporation
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
import json
import sys
from argparse import ArgumentParser


def main(argv):
    parser = ArgumentParser()
    parser.add_argument('--autoq-file', help='Path to input file', required=True)
    parser.add_argument('--hawq-file', help='Path to output file', required=False)
    args = parser.parse_args(args=argv)

    autoq_bitwidth_per_scope = []
    with open(args.autoq_file) as f:
        list_lines = f.readlines()
        for line in list_lines:
            line = line.replace('<= 8 ', '').replace('<= 4 ', '')
            s = line.split(' ')
            a_bitwidth = s[0]
            a_scope = s[3].split('-')[1].replace('\n', '')


            if 'module_weight' in a_scope:
                a_scope = a_scope.replace('module_weight', '')
                a_scope = f'TargetType.OPERATION_WITH_WEIGHTS {a_scope}'
            if '|OUTPUT' in a_scope:
                a_scope = a_scope.replace('|OUTPUT', '')
                a_scope = f'TargetType.OPERATOR_POST_HOOK {a_scope}'
            # print(a_bitwidth, a_scope)
            autoq_bitwidth_per_scope.append([int(a_bitwidth), a_scope])


        autoq_bitwidth_per_scope = list(sorted(autoq_bitwidth_per_scope, key=lambda x: x[1]))

    str_bw = [str(element) for element in autoq_bitwidth_per_scope]
    print('\n'.join(['\n\"bitwidth_per_scope\": [', ',\n'.join(str_bw), ']']))
    # with open(args.hawq_file) as f:
    #     h_json = json.load(f)
    #     hawq_bitwidth_per_scope = h_json['bitwidth_per_scope']


if __name__ == '__main__':
    main(sys.argv[1:])
