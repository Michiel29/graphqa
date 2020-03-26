import os
from collections import deque
import argparse
from tqdm import trange

import numpy as np
import torch

from fairseq.data import Dictionary, indexed_dataset
from fairseq.data.data_utils import load_indexed_dataset


def main(args):
    text_data, annotation_data = load_text_annotations(args.data_path, args.prefix)

    sentence_end_pos = text_data.sizes.cumsum()
    sentence_start_pos = np.roll(sentence_end_pos, shift=1)
    sentence_start_pos[0] = 0
    setattr(text_data, 'sentence_start_pos', sentence_start_pos)
    setattr(text_data, 'sentence_end_pos', sentence_end_pos)

    n_entities = len(Dictionary.load(os.path.join(args.data_path, 'entity.dict.txt')))
    print('-- {} entities'.format(n_entities))

    edges = create_graph(
        text_data,
        annotation_data,
        n_entities,
        args.document_sep_len,
        args.max_entity_pair_distance,
        args.max_positions,
    )

    graph_path = os.path.join(args.data_path, args.prefix + '.graph')
    graph_builder = indexed_dataset.make_builder(
        graph_path + '.bin',
        impl='mmap',
        vocab_size=n_entities,
    )
    with trange(n_entities, desc='Building graph dataset') as progress_bar:
        for entity in progress_bar:
            edges[entity].sort()
            graph_builder.add_item(torch.IntTensor(edges[entity]))
    graph_builder.finalize(graph_path + '.idx')


def load_text_annotations(path, prefix):
    text_data =  load_indexed_dataset(
        os.path.join(path, prefix + '.text'),
        None,
        dataset_impl='mmap',
    )
    assert text_data is not None

    annotation_data = np.load(os.path.join(path, prefix + '.annotations.npy'))
    assert annotation_data is not None
    return text_data, annotation_data


def create_graph(
    text_data,
    annotation_data,
    n_entities,
    document_sep_len,
    max_entity_pair_distance,
    max_positions,
):
    edges = [list() for entity in range(n_entities)]
    # mentions ordered by starting position
    left_annotations = deque()
    num_undirected_edges = 0
    current_sentence, current_document = None, None
    edge_same_sentence = 0

    # annotation = (global starting position, global ending position, sentence idx, document idx, entity idx)
    with trange(len(annotation_data), desc='Collecting entity pairs') as progress_bar:
        for index in progress_bar:
            annotation = annotation_data[index]
            if annotation[2] == current_sentence:
                right_start_pos = annotation[0]
                right_end_pos = annotation[1]
                right_sentence_idx = annotation[2]
                right_entity = annotation[4]

                while (
                    len(left_annotations) > 0
                    and left_annotations[0][1] + max_entity_pair_distance <= right_start_pos
                ):
                    left_annotations.popleft()

                for left_annotation in left_annotations:
                    left_start_pos = left_annotation[0]
                    left_end_pos = left_annotation[1]
                    left_sentence_idx = left_annotation[2]
                    left_entity = left_annotation[4]

                    if left_entity != right_entity and right_end_pos - left_start_pos <= max_positions:
                        start_block, end_block = expand_mention(
                            text_data=text_data,
                            left_sentence_idx=left_sentence_idx,
                            left_start_pos=left_start_pos,
                            left_end_pos=left_end_pos,
                            right_sentence_idx=right_sentence_idx,
                            right_start_pos=right_start_pos,
                            right_end_pos=right_end_pos,
                            document_sep_len=document_sep_len,
                            max_positions=max_positions,
                        )
                        edges[left_entity].append(
                            (right_entity, left_entity, left_start_pos, left_end_pos, right_start_pos, right_end_pos, start_block, end_block)
                        )
                        edges[right_entity].append(
                            (left_entity, right_entity, right_start_pos, right_end_pos, left_start_pos, left_end_pos, start_block, end_block)
                        )
                        num_undirected_edges += 1
                        edge_same_sentence += int(left_sentence_idx == right_sentence_idx)
            else:
                left_annotations.clear()
                current_sentence = annotation[2]
                current_document = annotation[3]

            left_annotations.append(annotation)

            if index % 5000 == 0:
                progress_bar.set_postfix(
                    num_documents=current_document,
                    num_sentences=current_sentence,
                    num_undir_edges=num_undirected_edges,
                    num_undir_edges_same_s=edge_same_sentence,
                    entities_queue_sz=len(left_annotations),
                )

        progress_bar.set_postfix(
            num_documents=current_document,
            num_sentences=current_sentence,
            num_undir_edges=num_undirected_edges,
            num_undir_edges_same_s=edge_same_sentence,
            entities_queue_sz=len(left_annotations),
        )
        print('-- num documents %d, num sentences %d, num undirected edges %d (within same sentence %d)' % (
            current_document,
            current_sentence,
            num_undirected_edges,
            edge_same_sentence,
        ))

    return edges


def expand_mention(
    text_data,
    left_sentence_idx,
    left_start_pos,
    left_end_pos,
    right_sentence_idx,
    right_start_pos,
    right_end_pos,
    document_sep_len,
    max_positions,
):
    start_pos, end_pos = left_start_pos, right_end_pos
    if text_data.sentence_end_pos[right_sentence_idx] - text_data.sentence_start_pos[left_sentence_idx] <= max_positions:
        start_pos = text_data.sentence_start_pos[left_sentence_idx]
        end_pos = text_data.sentence_end_pos[right_sentence_idx]
    else:
        min_symm_expand = min(
            text_data.sentence_end_pos[right_sentence_idx] - end_pos,
            start_pos - text_data.sentence_start_pos[left_sentence_idx],
            int((max_positions - (end_pos - start_pos)) / 2),
        )
        start_pos -= min_symm_expand
        end_pos += min_symm_expand
        budget_left = max_positions - (end_pos - start_pos)
        if budget_left > 1:
            if start_pos == text_data.sentence_start_pos[left_sentence_idx]:
                end_pos += budget_left
            elif text_data.sentence_end_pos[right_sentence_idx] == end_pos:
                start_pos -= budget_left
            else:
                raise Exception('Something is bad')
        if end_pos - start_pos > max_positions:
            import pdb; pdb.set_trace()
            pass
        assert end_pos - start_pos <= max_positions
        assert start_pos >= text_data.sentence_start_pos[left_sentence_idx]
        assert end_pos <= text_data.sentence_end_pos[right_sentence_idx]
        return start_pos, end_pos

    while end_pos - start_pos < max_positions:
        if (
            left_start_pos - start_pos <= end_pos - right_end_pos
            and (left_sentence_idx > 0)
            and (end_pos - start_pos + text_data.sizes[left_sentence_idx - 1] <= max_positions)
            and (text_data.sizes[left_sentence_idx - 1] > document_sep_len)
        ):
            left_sentence_idx -= 1
            start_pos = text_data.sentence_start_pos[left_sentence_idx]
        elif (
            (right_sentence_idx + 1 < len(text_data))
            and (end_pos - start_pos + text_data.sizes[right_sentence_idx + 1] <= max_positions)
            and (text_data.sizes[right_sentence_idx + 1] > document_sep_len)
        ):
            right_sentence_idx += 1
            end_pos = text_data.sentence_end_pos[right_sentence_idx]
        else:
            break

    # May be want to expand to not full sentences later
    assert end_pos - start_pos <= max_positions
    assert start_pos >= text_data.sentence_start_pos[left_sentence_idx]
    assert end_pos <= text_data.sentence_end_pos[right_sentence_idx]
    return start_pos, end_pos


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Construction of IndexedDatasets for graph neighbors and edges')
    parser.add_argument('--data-path', type=str, help='Data directory', default='../data/bin_sample')
    parser.add_argument('--prefix', type=str)
    parser.add_argument('--document-sep-len', type=int)
    parser.add_argument('--max-positions', type=int)
    parser.add_argument('--max-entity-pair-distance', type=int)

    args = parser.parse_args()
    main(args)
