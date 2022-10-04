import torch
import random
import multiprocessing
import shlex
import json

import torch.nn as nn
import numpy as np

from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from typing import List, Dict, Tuple
from collections import defaultdict
from subprocess import Popen, PIPE

def seed_everything(seed: int = 42):
    """
    Seeds all the random generators in order to make the training process reproducible
    
    :param seed: seed value
    
    :return: None
    """
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_optimizer_and_scheduler(model: nn.Module, optimizer_config: Dict,
                                scheduler_config: Dict) -> Tuple[AdamW, torch.optim.lr_scheduler.LambdaLR]:
    """
    Creates AdamW optimizer and a linear scheduler with warmup given config

    :param model: nn.Module object - model, which should be trained
    :param optimizer_config: dictionary containing optimizer config (just learning rate for now)
    :param scheduler_config: dictionary containing scheduler config (just num_warmup_steps and
           num_training_steps for now)

    :return: optimizer: AdamW optimizer for the given model
             scheduler: constant scheduler with warmup for the created optimizer
    """

    named_parameters = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in named_parameters if not 'bias' in n],
         'weight_decay': 0.01},
        {'params': [p for n, p in named_parameters if 'bias' in n], 'weight_decay': 0.0}]

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=optimizer_config['learning_rate'])

    # scheduler = get_linear_schedule_with_warmup(optimizer, scheduler_config['num_warmup_steps'],
    #                                             scheduler_config['num_training_steps'])
    
    scheduler = get_constant_schedule_with_warmup(optimizer, scheduler_config['num_warmup_steps'])
    

    return optimizer, scheduler


def loss_fn(preds: List[torch.Tensor], labels: List[torch.LongTensor], ignore_index: int = -1) -> Tuple:
    """
    Loss function for training a QA model on Google's NQ. Corresponds to description given in
    Alberti et al (2019) (https://arxiv.org/pdf/1901.08634.pdf)

    :param preds: list of tensors containing predicted spans and answer types
    :param labels: list of tensors containing original spans and answer types
    :param ignore_index: specifies a target value that is ignored and does not contribute to the input gradient

    :return: start_loss: CrossEntropyLoss calculated for the provided start scores
             end_loss: CrossEntropyLoss calculated for the provided end scores
             class_loss: CrossEntropyLoss calculated for the provided class scores
    """

    start_preds, end_preds, class_preds = preds
    start_labels, end_labels, class_labels = labels

    start_loss = nn.CrossEntropyLoss(
        ignore_index=ignore_index)(start_preds, start_labels)
    end_loss = nn.CrossEntropyLoss(
        ignore_index=ignore_index)(end_preds, end_labels)
    class_loss = nn.CrossEntropyLoss()(class_preds, class_labels)

    return start_loss, end_loss, class_loss


def is_valid_span(sample: Dict, span: List[int]) -> bool:
    """
    Checks predicted span for being in some feasible borders (whether it contains concatenated
    question and whether start >= end)

    :param sample: dictionary containing annotations, context, concatenated sequence and token
           indices mapping
    :param span: predicted answer span

    :return: bool indicating whether the span lies within feasible borders
    """

    start_index, end_index = span
    if start_index > end_index:
        return False
    if start_index <= sample['question_len'] + 2:
        return False
    if start_index == end_index:
        return False
    return True


class NQPredictions(object):
    """
    Stores predictions as well as preprocessed data used for predicting.
    """

    def __init__(self):
        self.predictions = {}
        self.results = {}
        self.best_scores = defaultdict(float)

    def add_predictions(self, samples: List[Dict], scores: np.ndarray, spans: np.ndarray, class_preds: np.ndarray):
        """
        Adds batched predictions to internal storage

        :param samples: list of dictionaries containing annotations, context, concatenated
               sequence and token indices mapping
        :param scores: np.ndarray containing scores of each prediction in the batch
        :param spans: np.ndarray containing pairs of start and end indiсes of each
               prediction in the batch
        :param class_preds: тp.ndarray containing class prediction scores of each
               item in the batch

        :return: None
        """

        for i, sample in enumerate(samples):

            if is_valid_span(sample, spans[i]) and self.best_scores[sample['example_id']] < scores[i]:
                self.best_scores[sample['example_id']] = scores[i]
                self.predictions[sample['example_id']] = sample
                self.results[sample['example_id']] = [sample['doc_start'], spans[i], class_preds[i]]

            elif not is_valid_span(sample, spans[i]) and self.best_scores[sample['example_id']] == 0.:
                self.best_scores[sample['example_id']] = scores[i]
                self.predictions[sample['example_id']] = sample
                self.results[sample['example_id']] = [sample['doc_start'], np.array([-1, -1]), class_preds[i]]

    def transform_predictions(self, postprocessing: bool = True) -> Dict:
        """
        Transforms predictions to the form needed for the official Google's NQ Dataset evaluation
        script.Then postprocesses predictions using candidate_answers

        Prediction format:
        {'predictions': [
            {
              'example_id': -2226525965842375672,
              'long_answer': {
                'start_byte': 62657, 'end_byte': 64776,
                'start_token': 391, 'end_token': 604
              },
              'long_answer_score': 13.5,
              'short_answers': [
                {'start_byte': 64206, 'end_byte': 64280,
                 'start_token': 555, 'end_token': 560}, ...],
              'short_answers_score': 26.4,
              'yes_no_answer': 'NONE'
            }, ... ]
          }

        :param postprocessing: whether to postprocess predictions or not

        :return: prediction_dict: dictionary that contains predictions; ready to be saved and used
                 with official Google's NQ Dataset evaluation script
        """

        prediction_dict = {'predictions': []}

        for example_id in self.results.keys():
            doc_start, index, _ = self.results[example_id]
            sample = self.predictions[example_id]

            # getting word indices from predicted token indices

            tokenized_to_original_index = sample['tokenized_to_original']
            raw_start_index = tokenized_to_original_index[doc_start + index[0] - 2 - sample['question_len']]
            raw_end_index = tokenized_to_original_index[doc_start + index[1] - 2 - sample['question_len']]

            # resolving potential equality of start and end word indices after token to word mapping

            if tokenized_to_original_index[doc_start + index[1] - 2 - sample['question_len']] == \
                    tokenized_to_original_index[doc_start + index[0] - 2 - sample['question_len']]:
                raw_start_index = -1
                raw_end_index = -1

            final_start_index = -1 if postprocessing else raw_start_index
            final_end_index = -1 if postprocessing else raw_end_index
            final_score = str(-100) if postprocessing else str(self.best_scores[example_id])
            
            
            if postprocessing:

                for candidate in sample['candidates']:
                    if candidate['start_token'] <= raw_start_index and raw_end_index <= candidate['end_token']:
                        final_start_index = candidate['start_token']
                        final_end_index = candidate['end_token']
                        final_score = str(self.best_scores[example_id])
                    
                        break

            # constructing resulting prediction dictionary

            long_dict = {'start_token': final_start_index, 'end_token': final_end_index, 'start_byte': -1,
                         'end_byte': -1}

            pred_dict = {'example_id': example_id, 'long_answer': long_dict,
                         'long_answer_score': final_score, 'short_answers': [],
                         'short_answers_score': 0, 'yes_no_answer': 'NONE'}

            prediction_dict['predictions'].append(pred_dict)

        return prediction_dict


def run_evaluation(gold_path: str, predictions_path: str) -> Dict:
    """
    This function launches official Google's NQ Dataset evaluation script via subprocess library.
    
    :param gold_path: glob path to the directory, that contains original GT labels in .jsonl.gz format,
                      or path to a specific file that contains original GT labels in .jsonl.gz format
    :param predictions_path: path to .jsonl file that contains predictions in a following format:

    :return: dictionary, containing metrics for long answer and short answer predictions
    """
    
    cmd = f'python nq_eval.py --gold_path={gold_path} --predictions_path={predictions_path}'
    
    process = Popen(shlex.split(cmd), stdout=PIPE)
    (output, err) = process.communicate()
    exit_code = process.wait()
    
    metrics = json.loads(output)
    print(metrics)

    return metrics
