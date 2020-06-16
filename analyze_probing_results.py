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

print('\nn_pos_rules={}, n_strong_neg_rules={}, n_weak_neg_rules={}\n'.format(len(tacred_rules), len(strong_neg_rules), n_rules-len(tacred_rules)-len(strong_neg_rules)))

for category in ['pos', 'strong_neg', 'weak_neg']:
    print('{}: mean_score={:.4f}, mean_rank={:.2f}'.format(category, mean(agg_results[category]['scores']), mean(agg_results[category]['ranks'])))
print()

for group in ['top_100', 'top_500', 'top_1000', 'bottom_1000', 'bottom_500', 'bottom_100']:
    print('{}: n_pos={}, n_strong_neg={}, n_weak_neg={}'.format(group, len(agg_results[group]['pos']['scores']), len(agg_results[group]['strong_neg']['scores']), len(agg_results[group]['weak_neg']['scores'])))
    for category in ['pos', 'strong_neg', 'weak_neg']:
        print('{}, {}: mean_score={:.4f}'.format(group, category, mean(agg_results[group][category]['scores'])))
    print()