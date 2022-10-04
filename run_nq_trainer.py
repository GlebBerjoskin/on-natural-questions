import torch
import datetime
import wandb
import os
import functools
import json
import sys

import torch.nn as nn

from transformers import AutoModel, AutoModelForQuestionAnswering, AutoTokenizer

from prepare_nq_data import prepare_nq_data
from nq_data_utils import train_collate_fn, eval_collate_fn
from nq_train_utils import get_optimizer_and_scheduler, loss_fn, seed_everything
from nq_model import NQModel
from nq_trainer import NQTrainer

if __name__ == '__main__':
    
    with open(sys.argv[1]) as file:
        CONFIG = json.load(file)
        
    if len(sys.argv) < 3:
        print('Provide config filename and usage scenario!\n'
              'E.g. :  python run_nq_trainer.py predict_config.json --predict\n'
              'E.g. :  python run_nq_trainer.py train_config.json --train')
        sys.exit()
    
    usage_scenario = sys.argv[2]
    
    if usage_scenario == '--train':
        
        num_training_steps = int(
        CONFIG['train_size'] / CONFIG['train_batch_size'] / CONFIG['gradient_accumulation_steps'])
        CONFIG['scheduler_config']['num_training_steps'] = num_training_steps

        now = datetime.datetime.now()
        RUN_NAME = f'{CONFIG["model_name"]}-{now.month}-{now.day}-{now.hour}-{now.minute}'
        
        seed_everything(CONFIG['seed'])
        
        print(f'Starting train run... \nRun name: {RUN_NAME}\n')

        # defining model

        default_qa_model = AutoModelForQuestionAnswering.from_pretrained(
            CONFIG['model_name'])
        default_cls_model = AutoModel.from_pretrained(CONFIG['model_name'])
        nq_model = NQModel(default_qa_model.roberta, default_qa_model.qa_outputs, default_cls_model.pooler,
                        num_labels=CONFIG['num_labels'])

        # sending model to device

        device = torch.device('cuda')
        # is_multi_gpu = torch.cuda.device_count() > 1
        # if is_multi_gpu:
        #     nq_model = nn.DataParallel(nq_model)
        nq_model.to(device)

        # defining tokenizer, optimizer and scheduler

        tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
        optimizer, scheduler = get_optimizer_and_scheduler(
            nq_model, CONFIG['optimizer_config'], CONFIG['scheduler_config'])

        # defining data utils

        eval_filenames = [filename for filename in os.listdir(CONFIG['test_data_path']) if
                        'dev' in filename and 'gz' not in filename]

        prepare_data_train = functools.partial(prepare_nq_data, tokenizer=tokenizer,
                                            class_label_mapping=CONFIG['class_label_mapping'],
                                            max_seq_len=CONFIG['max_seq_len'], max_quest_len=CONFIG['max_quest_len'],
                                            doc_stride=CONFIG['doc_stride'])

        prepare_data_eval = functools.partial(prepare_nq_data, tokenizer=tokenizer,
                                            class_label_mapping=CONFIG['class_label_mapping'],
                                            max_seq_len=CONFIG['max_seq_len'], max_quest_len=CONFIG['max_quest_len'],
                                            doc_stride=CONFIG['doc_stride'], test_scenario=True)

        # configuring WandB

        wandb.init(project=CONFIG['wandb_project'],
                entity=CONFIG['wandb_entity'], name=RUN_NAME, config=CONFIG)
        wandb.watch(nq_model, log_freq=CONFIG['wandb_watch_freq'])

        # training model

        trainer = NQTrainer(nq_model, loss_fn, optimizer,
                            scheduler, CONFIG, device, RUN_NAME, True)
        trainer.train(CONFIG['train_filenames'], eval_filenames, prepare_data_train,
                    prepare_data_eval, train_collate_fn, eval_collate_fn)
        
    if usage_scenario == '--predict':
        
        seed_everything(CONFIG['seed'])

        # defining model

        default_qa_model = AutoModelForQuestionAnswering.from_pretrained(CONFIG['model_name'])
        default_cls_model = AutoModel.from_pretrained(CONFIG['model_name'])
        nq_model = NQModel(default_qa_model.roberta, default_qa_model.qa_outputs, default_cls_model.pooler,
                        num_labels=CONFIG['num_labels'])
        
        # loading model from checkpoint
        
        checkpoint = torch.load(CONFIG['checkpoint_path'])
        nq_model.load_state_dict(checkpoint['model_state_dict'])
        
        # sending model to device

        device = torch.device('cuda')
        # is_multi_gpu = torch.cuda.device_count() > 1
        # if is_multi_gpu:
        #     nq_model = nn.DataParallel(nq_model)
        nq_model.to(device)

        # defining tokenizer

        tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
        
        # defining data utils

        eval_filenames = [filename for filename in os.listdir(CONFIG['test_data_path']) if
                        'dev' in filename and 'gz' not in filename]

        prepare_data_eval = functools.partial(prepare_nq_data, tokenizer=tokenizer,
                                            class_label_mapping=CONFIG['class_label_mapping'],
                                            max_seq_len=CONFIG['max_seq_len'], doc_stride=CONFIG['doc_stride'],
                                            max_quest_len=CONFIG['max_quest_len'], test_scenario=True)

        # predicting

        trainer = NQTrainer(nq_model, None, None, None, CONFIG, device, None, False)
        predictions = trainer.predict(eval_filenames, prepare_data_eval, eval_collate_fn)
        
        print('Saving predictions...')
        with open(CONFIG['predictions_path'], 'w') as file:
            json.dump(predictions, file)

    else:
        print('invalid usage scenario')
