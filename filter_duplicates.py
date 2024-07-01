# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from collections import OrderedDict

import pandas as pd

s = (
    "/home/nlyaly/projects/nncf2/tests/post_training/data/wwb_ref_answers/"
    "tinyllama__tinyllama-1.1b-step-50k-105b/ref_qa_old.csv"
)
data = pd.read_csv(s)
new_answers = []
for i, a in enumerate(data["answers"]):
    # print(i, ' before ', a)
    # text = text.replace('\n', '')
    # sentences = text.split('. ')
    # for i in range(len(sentences)):
    #     if sentences[i].startswith(' '):
    #         sentences[i] = sentences[i][1:]

    # sentences = list(OrderedDict.fromkeys(sentences))

    # if len(sentences) > 1:
    #     if sentences[-2].startswith(sentences[-1]):
    #         sentences = sentences[:-1]
    # text = '.'.join(sentences)
    # print(i, ' after ', text)

    sentences = re.split("\?|\!|\.", a)
    # print(*sentences, sep='\n')
    sentences = list(OrderedDict.fromkeys(sentences))
    # print(*sentences, sep='\n')
    if len(sentences) > 1:
        remove_last = False
        for s in sentences:
            if s.startswith(sentences[-1]):
                remove_last = True
                break
        if remove_last:
            sentences = sentences[:-1]
    new_answer = ".".join(sentences)
    # print(i, ' after ', new_answer, '\n\n')
    new_answers.append(new_answer)

d = {"questions": list(data["questions"]), "answers": new_answers}
s_new = (
    "/home/nlyaly/projects/nncf2/tests/post_training/data/wwb_ref_answers/"
    "tinyllama__tinyllama-1.1b-step-50k-105b/ref_qa_new.csv"
)
pd.DataFrame.from_dict(d).to_csv(s_new)
