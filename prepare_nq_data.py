import random
import re
import json

import pandas as pd

from typing import List, Dict, Tuple
from transformers import AutoTokenizer


def tokenize_nq_context(tokenizer: AutoTokenizer, context_words: List[str]) -> Tuple[List[str], List[int], List[int]]:
    """
    Tokenizes provided text, deletes html tags and creates mapping from original token numbering
    to model-used token numbering

    :param tokenizer: transformers.AutoTokenizer object, used for tokenization
    :param context_words: list of words that represent a html page from Google's NQ Dataset

    :return: context_tokens: list of tokens,
             original_to_tokenized: list representing mapping of original word indices to tokens
             tokenized_to_original: list representing mapping of tokens indices to original words
    """

    original_to_tokenized = []
    tokenized_to_original = []
    context_tokens = []

    for idx, word in enumerate(context_words):
        original_to_tokenized.append(len(context_tokens))

        # deleting html tags

        if re.match(r'<.+>', word):
            continue

        sub_tokens = tokenizer.tokenize(word)
        for sub_token in sub_tokens:
            tokenized_to_original.append(idx)
            context_tokens.append(sub_token)

    return context_tokens, original_to_tokenized, tokenized_to_original


def get_nq_annotations(row: Dict, original_to_tokenized: List[int], class_label_mapping: Dict) -> Tuple[int, int, int]:
    """
    Provides QA annotations for a given dictionary from Google's NQ Dataset. This function provides
    annotations for long answers and missing answers. This function ignores short answer annotations
    as wel as YES/NO answer annotations.

    :param row: dictionary representing one data entry from Google's NQ Dataset
    :param original_to_tokenized: list representing mapping of original word indices to tokens
    :param class_label_mapping: dictionary containing mapping of class names to integer values

    :return: class_label: mapping of the provided class name to a number
             start_position: index of the annotated answer start token
             end_position: index of the annotated answer end token
    """
    def _find_short_range(short_answers: List[Dict]) -> Tuple[int, int]:
        answers = pd.DataFrame(short_answers)
        start_min = answers['start_token'].min()
        end_max = answers['end_token'].max()
        return start_min, end_max
    
    if row['annotations'][0]['yes_no_answer'] in ['YES', 'NO']:
        class_label = row['annotations'][0]['yes_no_answer'].lower()
        start_position = row['annotations'][0]['long_answer']['start_token']
        end_position = row['annotations'][0]['long_answer']['end_token']
        
    elif row['annotations'][0]['short_answers']:
        class_label = 'short'
        start_position, end_position = _find_short_range(row['annotations'][0]['short_answers'])
        
    elif row['annotations'][0]['long_answer']['candidate_index'] != -1:
        class_label = 'long'
        start_position = row['annotations'][0]['long_answer']['start_token']
        end_position = row['annotations'][0]['long_answer']['end_token']
        
    else:
        class_label = 'none'
        start_position = -1
        end_position = -1
        
    # if row['annotations'][0]['long_answer']['start_token'] != -1:
    #     class_label = class_label_mapping['long']
    #     start_position = row['annotations'][0]['long_answer']['start_token']
    #     end_position = row['annotations'][0]['long_answer']['end_token']

    # else:
    #     class_label = class_label_mapping['none']
    #     start_position = -1
    #     end_position = -1

    # converting annotations to tokenized format

    if start_position != -1 and end_position != -1:
        start_position = original_to_tokenized[start_position]
        end_position = original_to_tokenized[end_position]

    return class_label, start_position, end_position


def prepare_nq_data(row: str, tokenizer: AutoTokenizer, class_label_mapping: Dict, max_seq_len: int, max_quest_len: int, doc_stride: int,
                    test_scenario: bool = False) -> List[Dict]:
    """
    Prepares Google's NQ Dataset for long QA model training. This function uses the same
    approach as Alberti et al (2019) (https://arxiv.org/pdf/1901.08634.pdf) with sliding
    window and [CLS]question[SEP]context[SEP] concatenation. This function ignores short
    answer annotations as wel as YES/NO answer annotations.

    :param row: string, representing single data entry from Google's NQ Dataset
    :param tokenizer: transformers.AutoTokenizer object, used for tokenization
    :param class_label_mapping: dictionary containing mapping of class names to
           integer values
    :param max_seq_len: max length of annotated sequence that can be generated
           during sliding window iterations through the context
    :param max_quest_len:
    :param doc_stride: num of tokens to skip while generating next sequence
           during sliding window iterations through the context
    :param test_scenario: boolean indicating whether we're dealing with train or test data

    :return: samples: list of dictionaries that contain annotations (if test_scenario==False)
             as well as context, resulting concatenated sequence and token indices mapping.
    """

    row = json.loads(row)
    context_words = [token_data['token'] for token_data in row['document_tokens']]
    context_tokens, original_to_tokenized, tokenized_to_original = tokenize_nq_context(tokenizer, context_words)
    question_tokens = tokenizer.tokenize(row['question_text'])[:max_quest_len]

    # obtaining annotations

    if not test_scenario:
        class_label, start_position, end_position = get_nq_annotations(row, original_to_tokenized, class_label_mapping)

    samples = []
    max_doc_len = max_seq_len - len(question_tokens) - 3  # [CLS], [SEP], [SEP]

    # obtaining document samples using sliding window technique

    for doc_start in range(0, len(context_tokens), doc_stride):
        doc_end = doc_start + max_doc_len

        if not test_scenario:
            if not (doc_start <= start_position and end_position <= doc_end):

                start, end, label = -1, -1, class_label_mapping['none']
            else:
                start = start_position - doc_start + len(question_tokens) + 2
                end = end_position - doc_start + len(question_tokens) + 2
                label = class_label_mapping[class_label]

        doc_tokens = context_tokens[doc_start:doc_end]
        input_tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + doc_tokens + ['[SEP]']
        if not test_scenario:
            samples.append({
                'example_id': row['example_id'],
                'annotations': row['annotations'],
                'candidates': row['long_answer_candidates'],
                'doc_start': doc_start,
                'tokenized_to_original': tokenized_to_original,
                'input_ids': tokenizer.convert_tokens_to_ids(input_tokens),
                'question_len': len(question_tokens),
                'start_position': start,
                'end_position': end,
                'class_label': label
            })
        else:
            samples.append({
                'example_id': row['example_id'],
                'candidates': row['long_answer_candidates'],
                'doc_start': doc_start,
                'tokenized_to_original': tokenized_to_original,
                'input_ids': tokenizer.convert_tokens_to_ids(input_tokens),
                'question_len': len(question_tokens)})

    return samples


def downsample_nq_samples(samples: List[List[Dict]]) -> List[Dict]:
    """
    A naive implementation of method that corrects the positive/negative distribution
    of a generated training set. It downsamples both empty (no answer) data entries and
    positive (containing answer) data entries, leaving just one data entry in the training
    set per original sample using random choice.

    :param samples: list of lists of dictionaries that contain annotations as well
           as context, resulting concatenated sequences and token indices mapping

    :return: downsampled_samples: list of dictionaries that contain annotations as well as context,
           resulting concatenated sequence and token indices mapping

    """

    downsampled_set = []
    n_long_samples = 0
    n_none_samples = 0
    none_label_indices = []

    # downsampling both negative and positive classes

    for idx, sample_set in enumerate(samples):
        annotated = list(filter(lambda sample: sample['class_label'] != 0, sample_set))

        if len(annotated) == 0:
            n_none_samples += 1
            none_label_indices.append(idx)

            downsampled_set.append(random.choice(sample_set))
        else:
            n_long_samples += 1
            downsampled_set.append(random.choice(annotated))

    # making sure negative samples won't overweight positive ones

    # n = n_none_samples - n_long_samples
    # if n:
    #     none_indices_to_delete = random.choices(range(len(none_label_indices)), k=n)
    #     for i, idx in enumerate(sorted(none_indices_to_delete)):
    #         downsampled_set.pop(idx-i)

    return downsampled_set
