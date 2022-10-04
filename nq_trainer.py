import numpy as np
import torch
import json
import wandb
import os
import itertools
import gc

import torch.nn as nn

from tqdm import tqdm
from typing import List, Dict, Tuple, Callable
from torch.utils.data import DataLoader

from prepare_nq_data import downsample_nq_samples
from nq_data_utils import NQDataset, NQJsonReader
from nq_train_utils import NQPredictions, run_evaluation


class NQTrainer:
    """
    Provides methods for training and evaluating QA model using Google's NQ Dataset
    """

    def __init__(self, model, loss_fn, optimizer, scheduler, config, device, run_name, log_metrics, global_step=0,
                 best_f1=-0):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.config = config
        self.device = device
        self.run_name = run_name
        self.log_metrics = log_metrics

        self.global_step = global_step
        self.best_f1 = best_f1

    def predict_iteration(self, inputs: List[torch.Tensor], samples: Dict) -> Tuple[
        Dict, np.ndarray, np.ndarray, np.ndarray]:
        """
        Runs one inference iteration using self.model and calculates prediction score as
        described in Alberti et al (2019) (https://arxiv.org/pdf/1901.08634.pdf)

        :param inputs:  list containing input_ids and attention_masks
        :param samples:  dictionary containing inf samples metadata

        :return: samples: dictionary containing inf samples metadata
                 scores: np.ndarray containing prediction scores
                 indices: np.ndarray containing predicted answer span indices
                 class_preds: np.ndarray containing predicted classes
        """
        # making_predictions

        input_ids, attention_mask = inputs
        predictions = self.model(input_ids.to(self.device), attention_mask.to(self.device))

        start_preds, end_preds, class_preds = (preds.detach().cpu() for preds in predictions)
        start_scores, start_index = torch.max(start_preds, dim=1)
        end_scores, end_index = torch.max(end_preds, dim=1)

        # calculating scores as stated in Alberti et al (2019) (https://arxiv.org/pdf/1901.08634.pdf)

        cls_logits = start_preds[:, 0] + end_preds[:, 0]
        scores = start_scores + end_scores - cls_logits
        indices = torch.stack((start_index, end_index)).transpose(0, 1)

        return samples, scores.numpy(), indices.numpy(), class_preds.numpy()
    
    def train_iteration(self, inputs: List[torch.Tensor], annotations: List[torch.LongTensor]):
        """
        Runs one train iteration using self.model and optionally logs loss components to WandB

        :param inputs: list of tensors containing input_ids and attention_masks
        :param annotations: list of tensors containing original spans and answer types

        :return: None
        """

        # making predictions

        input_ids, attention_mask = inputs
        predictions = self.model(input_ids.to(self.device), attention_mask=attention_mask.to(self.device))

        # calculating loss

        annotations = (ants.to(self.device) for ants in annotations)
        start_loss, end_loss, class_loss = self.loss_fn(predictions, annotations, ignore_index=-1)
        loss = start_loss + end_loss + class_loss

        loss.backward()
        
        # updating optimizer/scheduler

        if (self.global_step + 1) % self.config['gradient_accumulation_steps'] == 0:
            self.optimizer.step()
            self.scheduler.step()
            self.model.zero_grad()

        if self.log_metrics:
            log_dict = {}
            log_dict['train/batch_loss'] = loss.item()
            log_dict['train/batch_start_loss'] = start_loss.item()
            log_dict['train/batch_end_loss'] = end_loss.item()
            log_dict['train/batch_class_loss'] = class_loss.item()
            log_dict['train/lr'] = self.get_lr()
            wandb.log(log_dict)
                    

    def get_lr(self) -> float:
        """
        Returns current learning rate of self.optimizer

        :return: lr, current learning rate of self.optimizer
        """
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def predict(self, filenames: List[str], prepare_data_inf: Callable, inf_collate_fn: Callable) -> Dict:
        """
        Using provided filenames prepares Google's NQ Dataset data for inference (data
        is generated using sliding window scheme). Runs prediction process and then
        transforms predictions for them to be able to be used with official Google's
        NQ Dataset evaluation script.

        :param filenames: list of filenames that require predictions
        :param prepare_data_inf: callable object that describes inference data preparation
               procedure
        :param inf_collate_fn: collate_fn used to create DataLoaders from prepared inference
               data

        :return: predictions: dictionary that contains predictions; ready to be saved and used
                 with official Google's NQ Dataset evaluation script
        """

        self.model.eval()
        nq_predictions = NQPredictions()

        # iterating over files

        for filename in filenames:
            print(f'Predicting for file: {filename}...')

            data_reader = NQJsonReader(os.path.join(self.config['test_data_path'], filename),
                                       prepare_data_inf, chunksize=self.config['jsonlines_chunk_size'])

            # iterating over chunks of lines in a file

            for sample_batch in data_reader:
                samples = list(itertools.chain.from_iterable(sample_batch))
                val_dataset = NQDataset(samples)
                val_dataloader = DataLoader(val_dataset, batch_size=self.config['eval_batch_size'],
                                            collate_fn=inf_collate_fn)

                # iterating over batches in a chunk

                with torch.no_grad():
                    for inputs, examples in tqdm(val_dataloader, leave=True, position=0):
                        predictions = self.predict_iteration(inputs, examples)
                        nq_predictions.add_predictions(*predictions)
            

        torch.cuda.empty_cache()
        gc.collect()
        
        self.model.train()

        # formatting predictions as per Google's official evaluation script format

        print('Formatting predictions...')
        predictions = nq_predictions.transform_predictions()

        return predictions

    def evaluate(self, test_filenames: List[str], prepare_data_test: Callable, eval_collate_fn: Callable) -> Dict:
        """
        Using provided filenames runs prediction process and launches official Google's NQ Dataset
        evaluation script for obtained predictions

        :param test_filenames: list of filenames that are used during evaluation
        :param prepare_data_test: callable object that describes evaluation data preparation
               procedure
        :param eval_collate_fn: collate_fn used to create DataLoaders from prepared inference
               data

        :return: metrics: dictionary, containing metrics for long answer and short answer predictions
        """

        print('Making predictions...')
        predictions = self.predict(test_filenames, prepare_data_test, eval_collate_fn)
        with open(self.config['predictions_path'], 'w') as file:
            json.dump(predictions, file)

        print('Evaluating predictions...')
        metrics = run_evaluation(f'{self.config["test_data_path"]}/*.gz', self.config['predictions_path'])

        if self.log_metrics:
            log_dict = {}
            log_dict['eval/long-best-threshold-f1'] = metrics['long-best-threshold-f1']
            log_dict['eval/long-best-threshold-precision'] = metrics['long-best-threshold-precision']
            log_dict['eval/long-best-threshold-recall'] = metrics['long-best-threshold-recall']
            log_dict['eval/long-best-threshold'] = metrics['long-best-threshold']
            wandb.log(log_dict)

        return metrics

    def make_checkpoint(self, f_idx: int, filename: str, checkpoint_path: str):
        """
        Saves state_dicts of the model, the optimizer, the scheduler, as long as global_step and
        best_f1 of the training procedure

        :param f_idx: index of file that is used in train loop at the moment
        :param filename: name of file that is used in train loop at the moment
        :param checkpoint_path: where to save the checkpoint

        :return: None
        """
        print('Saving checkpoint...')

        torch.save({'global_step': self.global_step,
                    'filename': filename,
                    'f_idx': f_idx,
                    'best_f1': self.best_f1,

                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict()},
                   checkpoint_path)

        print('Checkpoint saved!')

    def train(self, train_filenames: List[str], test_filenames: List[str], prepare_data_train: Callable,
              prepare_data_eval: Callable, train_collate_fn: Callable, eval_collate_fn: Callable):
        """
        Using provided filenames runs training process and performs evaluation every
        self.config['eval_steps'] step, saving best model to ./models folder as well as
        corresponding optimizer and scheduler

        :param train_filenames: list of filenames that are used during training
        :param test_filenames: list of filenames that are used during evaluation
        :param prepare_data_train: callable object that describes training data preparation
               procedure
        :param prepare_data_eval: callable object that describes evaluation data preparation
               procedure
        :param train_collate_fn: collate_fn used to create DataLoaders from prepared train
               data
        :param eval_collate_fn: collate_fn used to create DataLoaders from prepared inference
               data

        :return: None
        """
        # iterating over files
        f_idx = 0

        for filename in train_filenames:
            print(f'Training on file: {filename}...')

            data_reader = NQJsonReader(os.path.join(self.config['train_data_path'], filename),
                                       prepare_data_train, chunksize=self.config['jsonlines_chunk_size'])

            # iterating over chunks of lines in a file

            for sample_batch in data_reader:
                samples = downsample_nq_samples(sample_batch)
                train_dataset = NQDataset(samples)
                train_dataloader = DataLoader(train_dataset, batch_size=self.config['train_batch_size'],
                                              shuffle=True, collate_fn=train_collate_fn)

                # iterating over batches in a chunk

                try:

                    for inputs, annotations in tqdm(train_dataloader, leave=True, position=0):
                        self.train_iteration(inputs, annotations)

                        # launching evaluation process

                        if (self.global_step + 1) % self.config['eval_steps'] == 0:
                            print('Evaluating...')

                            metrics = self.evaluate(test_filenames, prepare_data_eval, eval_collate_fn)
                            print(f'Best threshold F1: {metrics["long-best-threshold-f1"]}, '
                                  f'Best threshold: {metrics["long-best-threshold"]}')

                            # saving new best model

                            if metrics['long-best-threshold-f1'] > self.best_f1 or not self.best_f1:
                                print(
                                    f'New best F1 reached! New best F1 is '
                                    f'{metrics["long-best-threshold-f1"] - self.best_f1} more than the previous one')

                                self.best_f1 = metrics['long-best-threshold-f1']
                                self.make_checkpoint(f_idx, filename, os.path.join(self.config['model_path'],
                                                                                   self.run_name + '.pt'))

                        self.global_step += 1

                except Exception as e:
                    print(e)
                    self.make_checkpoint(f_idx, filename, os.path.join(self.config['model_path'],
                                                                       self.run_name + '_SAVED_CRASH' + '.pt'))
                    # break

            f_idx += 1

        # launching evaluation process

        print('Training for 1 epoch has finished! Evaluating...')

        metrics = self.evaluate(test_filenames, prepare_data_eval, eval_collate_fn)
        print(
            f'Best threshold F1: {metrics["long-best-threshold-f1"]}, '
            f'Best threshold: {metrics["long-best-threshold"]}')
        
        # saving new best model

        if metrics['long-best-threshold-f1'] > self.best_f1 or not self.best_f1:
            print(
                f'New best F1 reached! New best F1 is '
                f'{metrics["long-best-threshold-f1"] - self.best_f1} more than the previous one')

            self.best_f1 = metrics['long-best-threshold-f1']
            self.make_checkpoint(f_idx, 'last_file_in_queue', os.path.join(self.config['model_path'],
                                                                           self.run_name + '.pt'))
        