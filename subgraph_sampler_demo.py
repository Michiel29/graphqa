import argparse
from collections import Counter
import copy
import numpy as np
import os
import pygraphviz as pgv
from tqdm import trange
import logging
from scipy.stats import wasserstein_distance

from datasets import AnnotatedText, GraphDataset, SubgraphSampler
from utils.data_utils import numpy_seed, safe_load_indexed_dataset
from utils.dictionary import CustomDictionary, EntityDictionary
from utils.diagnostic_utils import Diagnostic
from utils.numpy_utils import MMapNumpyArray


logger = logging.getLogger(__name__)


def name(entity_dictionary, index):
    s = entity_dictionary[index]
    if len(s) > 20:
        s = s[:20] + '...'
    return s + '\n' + str(index)


def save_subgraph(subgraph, dictionary, entity_dictionary, path, shall_add_text):
    if shall_add_text:
        diagnostic = Diagnostic(dictionary, entity_dictionary)

    G = pgv.AGraph(strict=True, directed=True)
    for (h, t), sentence in subgraph.relation_statements.items():
        if (h, t) in subgraph.covered_entity_pairs:
            color = 'red'
        else:
            color = 'black'

        if shall_add_text:
            text = diagnostic.decode_text(sentence)
            G.add_edge(name(entity_dictionary, h), name(entity_dictionary, t), color=color, label=text)
        else:
            G.add_edge(name(entity_dictionary, h), name(entity_dictionary, t), color=color)

    G.layout(prog='dot')
    G.draw(path + '.png')


def sample_subgraph(graph, annotated_text, index, entity_pair_counter, entity_pair_counter_sum, args):
    subgraph = SubgraphSampler(
        graph=graph,
        annotated_text=annotated_text,
        min_common_neighbors=args.min_common_neighbors,
        max_entities_size=args.max_entities_size,
        max_entities_from_queue=args.max_entities_from_queue,
        cover_random_prob=args.cover_random_prob,
        entity_pair_counter=copy.deepcopy(entity_pair_counter),
        entity_pair_counter_sum=entity_pair_counter_sum,
        entity_pair_counter_cap=args.entity_pair_counter_cap,
    )
    edge = graph[index]
    head = edge[GraphDataset.HEAD_ENTITY]
    tail = edge[GraphDataset.TAIL_ENTITY]
    sentence = annotated_text.annotate_relation(*(edge.numpy()))

    if not subgraph.add_initial_entity_pair(head, tail, args.max_tokens, args.max_sentences, sentence):
        return None, None

    result = subgraph.fill(
        args.max_tokens,
        args.max_sentences,
        args.min_common_neighbors_for_the_last_edge)
    return subgraph, result


def main(args):
    dict_path = os.path.join(args.data_path, 'dict.txt')
    dictionary = CustomDictionary.load(dict_path)

    entity_dict_path = os.path.join(args.data_path, 'entity.dict.txt')
    entity_dictionary = EntityDictionary.load(entity_dict_path)

    logger.info('dictionary: {} types'.format(len(dictionary)))
    logger.info('entity dictionary: {} types'.format(len(entity_dictionary)))

    text_data = safe_load_indexed_dataset(
        os.path.join(args.data_path, args.split + '.text'),
    )
    annotation_data = MMapNumpyArray(
        os.path.join(args.data_path, args.split + '.annotations.npy')
    )
    annotated_text = AnnotatedText(
        text_data=text_data,
        annotation_data=annotation_data,
        dictionary=dictionary,
        mask_type=args.mask_type,
        non_mask_rate=args.non_mask_rate,
    )

    graph_data = safe_load_indexed_dataset(
        os.path.join(args.data_path, args.split + '.graph'),
    )
    graph = GraphDataset(
        edges=graph_data,
        subsampling_strategy=args.subsampling_strategy,
        subsampling_cap=args.subsampling_cap,
        seed=args.seed,
    )
    graph.set_epoch(1)

    entity_pair_counter_sum = 0

    with numpy_seed('SubgraphSampler', args.seed):
        random_perm = np.random.permutation(len(graph))

        if args.save_subgraph is not None:
            for index in random_perm:
                subgraph, _ = sample_subgraph(graph, annotated_text, index, None, None, args)
                if subgraph is not None:
                    break

            path = '%s.max_tokens=%d.max_sentences=%d.min_common_neighbors=%d' % (
                args.save_subgraph,
                args.max_tokens,
                args.max_sentences,
                args.min_common_neighbors,
            )
            save_subgraph(subgraph, dictionary, entity_dictionary, path, args.save_text)
        else:
            num_subgraphs, total_edges, total_covered_edges = 0, 0, 0
            total_relative_coverage_mean, total_relative_coverage_median = 0, 0
            total_full_batch = 0
            entity_pair_counter, relation_statement_counter = Counter(), Counter()
            with trange(len(graph), desc='Sampling') as progress_bar:
                for i in progress_bar:
                    subgraph, fill_successfully = sample_subgraph(
                        graph,
                        annotated_text,
                        random_perm[i],
                        entity_pair_counter,
                        entity_pair_counter_sum,
                        args,
                    )
                    if subgraph is None:
                        continue

                    num_subgraphs += 1
                    relation_statement_counter.update([hash(x) for x in subgraph.relation_statements.values()])
                    # entity_pair_counter.update([(min(h, t), max(h, t)) for (h, t) in subgraph.relation_statements.keys()])
                    entity_pair_counter.update([(h, t) for (h, t) in subgraph.relation_statements.keys()])
                    entity_pair_counter_sum += len(subgraph.relation_statements)
                    total_edges += len(subgraph.relation_statements)
                    total_covered_edges += len(subgraph.covered_entity_pairs)
                    relative_coverages = subgraph.relative_coverages()
                    total_relative_coverage_mean += np.mean(relative_coverages)
                    total_relative_coverage_median += np.median(relative_coverages)
                    total_full_batch += int(fill_successfully)

                    entity_pairs_counts = np.array(list(entity_pair_counter.values()))
                    relation_statement_counts = np.array(list(relation_statement_counter.values()))

                    progress_bar.set_postfix(
                        # n=num_subgraphs,
                        mean=entity_pair_counter_sum / len(graph),
                        m_r=relation_statement_counter.most_common(1)[0][1],
                        m_e=entity_pair_counter.most_common(1)[0][1],
                        w_e=wasserstein_distance(entity_pairs_counts, np.ones_like(entity_pairs_counts)),
                        w_r=wasserstein_distance(relation_statement_counts, np.ones_like(relation_statement_counts)),
                        y=total_covered_edges / total_edges,
                        e=total_edges / num_subgraphs,
                        # cov_e=total_covered_edges / num_subgraphs,
                        rel_cov=total_relative_coverage_mean / num_subgraphs,
                        # rel_cov_median=total_relative_coverage_median / num_subgraphs,
                        # f=total_full_batch / num_subgraphs,
                    )


def cli_main():
    parser = argparse.ArgumentParser(description='Graph subsampling demo')
    parser.add_argument('--data-path', type=str, help='path to data directory')
    parser.add_argument('--split', type=str, default='train', help='which data split to use: train (default) or validation')
    parser.add_argument('--mask-type', type=str, default='start_end', help='Mask type for the text annotations')
    parser.add_argument('--non-mask-rate', type=float, default=1, help='How often we keep the entity\'s surface form')
    parser.add_argument('--cover-random-prob', type=float, default=0)
    parser.add_argument('--subsampling-strategy', type=str, default='by_entity_pair')
    parser.add_argument('--subsampling-cap', type=int, default=1)
    parser.add_argument('--seed', type=int, default=31415)
    parser.add_argument('--min-common-neighbors', type=int, default=10)
    parser.add_argument('--min-common-neighbors-for-the-last-edge', type=int, default=1)
    parser.add_argument('--max-tokens', type=int, default=18000)
    parser.add_argument('--max-sentences', type=int, default=1000)
    parser.add_argument('--max-entities-size', type=int, default=600)
    parser.add_argument('--max-entities-from-queue', type=int, default=5)
    parser.add_argument('--entity-pair-counter-cap', type=int, default=100)
    parser.add_argument('--save-subgraph', type=str)
    parser.add_argument('--save-text', action='store_true', default=False)
    parsed_args = parser.parse_args()
    main(parsed_args)


if __name__ == '__main__':
    cli_main()