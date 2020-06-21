import os
from statistics import mean
from tqdm import tqdm
from utils.probing_utils import load_probing_results, tacred_relations, tacred_rules 

save_dir = '../data/tacred/tacred/data/bin/probing'
all_results_path = os.path.join(save_dir, 'all_results.pkl')
rule_results_path = os.path.join(save_dir, 'rule_results.pkl')
strong_neg_rules_path = os.path.join(save_dir, 'strong_neg_rules.pkl')

all_results = load_probing_results(all_results_path)
rule_results = load_probing_results(rule_results_path)
strong_neg_rules = load_probing_results(strong_neg_rules_path)
tacred_rules_ = [(tacred_relations[r1], tacred_relations[r2], tacred_relations[r3]) for (r1, r2, r3) in tacred_rules]
strong_neg_rules_ = [(tacred_relations[r1], tacred_relations[r2], tacred_relations[r3]) for (r1, r2, r3) in strong_neg_rules]
n_rules = len(all_results)
n_pos_rules = len(tacred_rules)
n_strong_neg_rules = len(strong_neg_rules)
n_weak_neg_rules = n_rules - n_pos_rules - n_strong_neg_rules

agg_results = {
    'pos': {'scores': [], 'ranks': []}, 
    'strong_neg': {'scores': [], 'ranks': []}, 
    'weak_neg': {'scores': [], 'ranks': []}, 
    'top_100': {
        'pos': {'scores': [], 'ranks': []},
        'strong_neg': {'scores': [], 'ranks': []},
        'weak_neg': {'scores': [], 'ranks': []},
    },
    'top_500': {
        'pos': {'scores': [], 'ranks': []},
        'strong_neg': {'scores': [], 'ranks': []},
        'weak_neg': {'scores': [], 'ranks': []},
    },
    'top_1000': {
        'pos': {'scores': [], 'ranks': []},
        'strong_neg': {'scores': [], 'ranks': []},
        'weak_neg': {'scores': [], 'ranks': []},
    },
    'bottom_1000': {
        'pos': {'scores': [], 'ranks': []},
        'strong_neg': {'scores': [], 'ranks': []},
        'weak_neg': {'scores': [], 'ranks': []},
    },
    'bottom_500': {
        'pos': {'scores': [], 'ranks': []},
        'strong_neg': {'scores': [], 'ranks': []},
        'weak_neg': {'scores': [], 'ranks': []},
    },
    'bottom_100': {
        'pos': {'scores': [], 'ranks': []},
        'strong_neg': {'scores': [], 'ranks': []},
        'weak_neg': {'scores': [], 'ranks': []},
    },
}

for rank in tqdm(range(len(all_results)), desc='Aggregating scores/ranks'):
    result = all_results[rank]
    if result['rule'] in tacred_rules_:
        category = 'pos'
    elif result['rule'] in strong_neg_rules_:
        category = 'strong_neg'
    else:
        category = 'weak_neg'
        
    agg_results[category]['scores'].append(result['mean_score'])
    agg_results[category]['ranks'].append(rank)

    if rank < 100:
        agg_results['top_100'][category]['scores'].append(result['mean_score'])
        agg_results['top_100'][category]['ranks'].append(rank)
    if rank < 500:
        agg_results['top_500'][category]['scores'].append(result['mean_score'])
        agg_results['top_500'][category]['ranks'].append(rank)
    if rank < 1000:
        agg_results['top_1000'][category]['scores'].append(result['mean_score'])
        agg_results['top_1000'][category]['ranks'].append(rank)
    if rank >= n_rules-1000:
        agg_results['bottom_1000'][category]['scores'].append(result['mean_score'])
        agg_results['bottom_1000'][category]['ranks'].append(rank)
    if rank >= n_rules-500:
        agg_results['bottom_500'][category]['scores'].append(result['mean_score'])
        agg_results['bottom_500'][category]['ranks'].append(rank)
    if rank >= n_rules-100:
        agg_results['bottom_100'][category]['scores'].append(result['mean_score'])
        agg_results['bottom_100'][category]['ranks'].append(rank)

print('\nn_pos_rules={}, n_strong_neg_rules={}, n_weak_neg_rules={}\n'.format(n_pos_rules, n_strong_neg_rules, n_weak_neg_rules))

for category in ['pos', 'strong_neg', 'weak_neg']:
    print('{}: mean_score={:.4f}, mean_rank={:.2f}'.format(category, mean(agg_results[category]['scores']), mean(agg_results[category]['ranks'])))
print()

for group in ['top_100', 'top_500', 'top_1000', 'bottom_1000', 'bottom_500', 'bottom_100']:
    print('{}: n_pos={} ({:.1f}%), n_strong_neg={} ({:.1f}%), n_weak_neg={} ({:.1f}%)'.format(group, len(agg_results[group]['pos']['scores']), len(agg_results[group]['pos']['scores']) / n_pos_rules * 100, len(agg_results[group]['strong_neg']['scores']), len(agg_results[group]['strong_neg']['scores']) / n_strong_neg_rules * 100, len(agg_results[group]['weak_neg']['scores']), len(agg_results[group]['weak_neg']['scores']) / n_weak_neg_rules * 100))
    for category in ['pos', 'strong_neg', 'weak_neg']:
        try:
            print('{}, {}: mean_score={:.4f}'.format(group, category, mean(agg_results[group][category]['scores'])))
        except:
            print('{}, {}: mean_score=N/A'.format(group, category))
    print()


for rank in tqdm(range(500), desc='Saving results of top 500 rules to file'):
    result = all_results[rank]
    target_relation = result['rule'][0]
    evidence_relations = result['rule'][1], result['rule'][2]
    if result['rule'] in tacred_rules_:
        category = 'pos'
    elif result['rule'] in strong_neg_rules_:
        category = 'strong_neg'
    else:
        category = 'weak_neg'

    with open(os.path.join(save_dir, 'top_500_results.txt'), 'a') as f:
        print('rule {}: target={}, evidence=({}, {}), mean_score={:.6f}, category={}'.format(rank, target_relation, evidence_relations[0], evidence_relations[1], result['mean_score'], category), file=f)
        print('first five samples (out of 100):', file=f)
        for i in range(5):
            cur_decoded_rule = result['decoded_rules'][i]
            print('- sample {}'.format(i), file=f)
            print('\t- target sentence: {}'.format(cur_decoded_rule['target'][target_relation]), file=f)
            print('\t- evidence sentence 1: {}'.format(cur_decoded_rule['evidence'][0][evidence_relations[0]]), file=f)
            print('\t- evidence sentence 2: {}'.format(cur_decoded_rule['evidence'][1][evidence_relations[1]]), file=f)
        print('\n', file=f)

for rank in tqdm(range(500), desc='Saving results of bottom 500 rules to file'):
    result = all_results[n_rules-rank-1]
    target_relation = result['rule'][0]
    evidence_relations = result['rule'][1], result['rule'][2]
    if result['rule'] in tacred_rules_:
        category = 'pos'
    elif result['rule'] in strong_neg_rules_:
        category = 'strong_neg'
    else:
        category = 'weak_neg'

    with open(os.path.join(save_dir, 'bottom_500_results.txt'), 'a') as f:
        print('rule {}: target={}, evidence=({}, {}), mean_score={:.6f}, category={}'.format(n_rules-rank, target_relation, evidence_relations[0], evidence_relations[1], result['mean_score'], category), file=f)
        print('first five samples (out of 100):', file=f)
        for i in range(5):
            cur_decoded_rule = result['decoded_rules'][i]
            print('- sample {}'.format(i), file=f)
            print('\t- target sentence: {}'.format(cur_decoded_rule['target'][target_relation]), file=f)
            print('\t- evidence sentence 1: {}'.format(cur_decoded_rule['evidence'][0][evidence_relations[0]]), file=f)
            print('\t- evidence sentence 2: {}'.format(cur_decoded_rule['evidence'][1][evidence_relations[1]]), file=f)
        print('\n', file=f)