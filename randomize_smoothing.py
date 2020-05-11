# -*- coding: utf-8 -*-
import argparse
from package.imports.universal import *
from package.utils.utils import *
from package.parse.perturb.unify import Parser_Unify

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    parser_unify = Parser_Unify(output=False)
    dataset = parser_unify.module.dataset
    model = parser_unify.module.model
    perturb = parser_unify.module.perturb
    m_dict = parser_unify.load_file()

    # ------------------------------------------------------------------------ #

    fail = 0
    total = 0
    org = 0
    target = 0
    benign_confidence_list = []
    org_confidence_list = []
    target_confidence_list = []
    for key in m_dict.keys():
        if total >= 100:
            break
        # print(key)
        model.load_pretrained_weights()
        X_org = to_tensor(m_dict[key]['X_org'])
        X_var = to_tensor(m_dict[key]['X_var'])
        label_org = to_tensor([m_dict[key]['label_org']])
        label_target = to_tensor([m_dict[key]['label_target']])

        _result1 = model.get_prob(X_org)
        _result2 = model.get_prob(X_org, randomized_smooth=True)
        benign_confidence_list.append(
            float(_result1.squeeze()[label_org])-float(_result2.squeeze()[label_org]))

        _iter = perturb.module.poison.perturb(X_var, label_target)
        _, acc, _ = model._validate(full=False, output=False)
        confidence = model.get_target_confidence(X_var, label_target)

        _result = model.get_prob(X_var)
        if float(_result.squeeze()[label_target]) < perturb.stop_confidence:
            # print('reconstruct model fail')
            fail += 1
            continue
        total += 1
        # print('  original class confidence: ',
        #         float(_result.squeeze()[label_org]))
        # print('  target class confidence: ',
        #         float(_result.squeeze()[label_target]))

        _result = model.get_prob(X_var, randomized_smooth=True)
        # print('  original randomized smooth confidence: ',
        #         float(_result.squeeze()[label_org]))
        # print('  target randomized smooth confidence: ',
        #         float(_result.squeeze()[label_target]))
        _confidence, _classification = _result.max(1)

        org_confidence_list.append(float(_result.squeeze()[label_org]))
        target_confidence_list.append(float(_result.squeeze()[label_target]))
        if _classification == label_org:
            org += 1
        elif _classification == label_target:
            target += 1

    print('fail / total: ', fail, '/', total)
    print('original proportion: ', float(org)/total)
    print('target proportion: ', float(target)/total)
    print('benign class confidence: ',
          np.array(benign_confidence_list).mean())
    print('original class confidence: ',
          np.array(org_confidence_list).mean())
    print('target class confidence: ', np.array(
        target_confidence_list).mean())
