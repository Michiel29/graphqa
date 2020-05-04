import argparse
import numpy as np
import os
import pygraphviz as pgv
from tqdm import trange
import logging

from datasets import AnnotatedText, GraphDataset, SubgraphSampler
from utils.data_utils import numpy_seed, safe_load_indexed_dataset
from utils.dictionary import CustomDictionary, EntityDictionary
from utils.numpy_utils import MMapNumpyArray


logger = logging.getLogger(__name__)


def name(entity_dictionary, index):
    s = entity_dictionary[index]
    if len(s) > 20:
        s = s[:20] + '...'
    return s


def save_subgraph(subgraph, entity_dictionary, path):
    G = pgv.AGraph(strict=True, directed=True)
    for h, t in subgraph.relation_statements.keys():
        assert (t, h) not in subgraph.covered_entity_pairs
        if (h, t) in subgraph.covered_entity_pairs:
            color = 'red'
        else:
            color = 'black'
        G.add_edge(name(entity_dictionary, h), name(entity_dictionary, t), color=color)
    G.write(path + '.dot')
    G.layout(prog='dot')
    G.draw(path + '.png')


def sample_subgraph(graph, annotated_text, index, args):
    subgraph = SubgraphSampler(
        graph=graph,
        annotated_text=annotated_text,
        min_common_neighbors=args.min_common_neighbors,
    )
    edge = graph[index]
    head = edge[GraphDataset.HEAD_ENTITY]
    tail = edge[GraphDataset.TAIL_ENTITY]
    sentence = annotated_text.annotate(*(edge.numpy()))

    if not subgraph.try_add_entity_pair_with_neighbors(head, tail, args.max_tokens, args.max_sentences, 1, sentence):
        return None, None

    result = subgraph.fill(args.max_tokens, args.max_sentences, args.min_common_neighbors_for_the_last_edge)

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

    subgraph = SubgraphSampler(
        graph=graph,
        annotated_text=annotated_text,
        min_common_neighbors=args.min_common_neighbors,
    )

    with numpy_seed('SubgraphSampler', args.seed):
        random_perm = np.random.permutation(len(graph))

    if args.save_subgraph is not None:
        subgraph = sample_subgraph(graph, annotated_text, random_perm[0], args)
        path = '%s.max_tokens=%d.max_sentences=%d.min_common_neighbors=%d' % (
            args.save_subgraph,
            args.max_tokens,
            args.max_sentences,
            args.min_common_neighbors,
        )
        save_subgraph(subgraph, entity_dictionary, path)
    else:
        num_subgraphs, total_edges, total_covered_edges = 0, 0, 0
        total_relative_coverage_mean, total_relative_coverage_median = 0, 0
        total_full_batch = 0
        with trange(len(graph), desc='Subgraph sampling') as progress_bar:
            for i in progress_bar:
                subgraph, fill_successfully = sample_subgraph(graph, annotated_text, random_perm[i], args)
                if subgraph is None:
                    continue

                num_subgraphs += 1
                total_edges += len(subgraph.relation_statements)
                total_covered_edges += len(subgraph.covered_entity_pairs)
                relative_coverages = subgraph.relative_coverages()
                total_relative_coverage_mean += np.mean(relative_coverages)
                total_relative_coverage_median += np.median(relative_coverages)
                total_full_batch += int(fill_successfully)

                progress_bar.set_postfix(
                    n=num_subgraphs,
                    edges=total_edges / num_subgraphs,
                    cov_edges=total_covered_edges / num_subgraphs,
                    rel_cov_mean=total_relative_coverage_mean / num_subgraphs,
                    rel_cov_median=total_relative_coverage_median / num_subgraphs,
                    full_batch=total_full_batch / num_subgraphs,
                )


def cli_main():
    parser = argparse.ArgumentParser(description='Graph subsampling demo')
    parser.add_argument('--data-path', type=str, help='path to data directory')
    parser.add_argument('--split', type=str, default='train', help='which data split to use: train (default) or validation')
    parser.add_argument('--mask-type', type=str, default='start_end', help='Mask type for the text annotations')
    parser.add_argument('--non-mask-rate', type=float, default=0, help='How often we keep the entity\'s surface form')
    parser.add_argument('--subsampling-strategy', type=str, default='by_entity_pair')
    parser.add_argument('--subsampling-cap', type=int, default=1)
    parser.add_argument('--seed', type=int, default=31415)
    parser.add_argument('--min-common-neighbors', type=int, default=10)
    parser.add_argument('--min-common-neighbors-for-the-last-edge', type=int, default=1)
    parser.add_argument('--max-tokens', type=int, default=2e4)
    parser.add_argument('--max-sentences', type=int, default=100000)
    parser.add_argument('--save-subgraph', type=str)
    parsed_args = parser.parse_args()
    main(parsed_args)


if __name__ == '__main__':
    cli_main()