 
## Getting started 
1.  Clone the repository, `cd` to repository root.   

3.  `cd` to **./data** directory and download original *train* data:
	 `gsutil -m cp -R gs://natural_questions/v1.0/train ./`
	 
4.  `cd` to **./data/train** directory and decompress *train* data:
	 `gunzip *.gz`

	 Running decompression in single-thread mode takes some time to complete. 
	 Decompressed version of the Google's NQ Dataset *train* requires ~270GB of free disk space.

5.  `cd` back to **./data** directory and download original *dev* data:
	 `gsutil -m cp -R gs://natural_questions/v1.0/dev ./`
	 
6.  `cd` to **./data/dev** directory and decompress *dev* data (we’ll need both `.jsonl` and `.jsonl.gz` files in future):
`gunzip -c nq-dev-00.jsonl.gz > nq-dev-00.jsonl`
`gunzip -c nq-dev-01.jsonl.gz > nq-dev-01.jsonl`
`gunzip -c nq-dev-02.jsonl.gz > nq-dev-02.jsonl`
`gunzip -c nq-dev-03.jsonl.gz > nq-dev-03.jsonl`
`gunzip -c nq-dev-04.jsonl.gz > nq-dev-04.jsonl`

7.  `cd` to repository root and use `requirements.txt` to install all the necessary libraries. Present `requirements.txt` has been used in *pyhon 3.8* / *CUDA 11.1 environment. Different *python* and *CUDA* versions may affect library versions that need to be installed.
  
8.  All is set up! 
To finetune a new RoBERTa model on NQ Dataset for 1 epoch, `cd` to repository root and run:
`python run_nq_trainer.py train_config.json --train`

	Use `train_config.json` to specify hyperparameters (e.g. data preparation params, or train params), or select which files to use for training/evaluation.

9. To make predictions using a checkpointed model, `cd` to repository root and run:
`python run_nq_trainer.py predict_config.json --predict`

	Use `predict_config.json` to configure your own parameters.
    
## Metrics
  The model, that can be loaded from **./models/roberta-large-9-15-8-43.pt** achieves following metrics while evaluating on  the long answer part of Google's NQ Dataset:
   * **F1**=0.478,
  *  **Precision**=0.470,
  *  **Recall**=0.486.
    
## How to replicate metrics (not applicable anymore, I had to delete large .pt file from this repository)

1.  Make sure the steps (1), (4), (5), (6) from [getting started](https://github.com/GlebBerjoskin/technical_assignment#getting-started) section are completed.

2.  `cd` to the repository root and run the prediction script:
	 `python run_nq_trainer.py predict_config.json --predict`
	
	Adjust eval_batch_size according to your machine specs.
	
3.  Run official evaluation script using
`python nq_eval.py --gold_path=./data/dev/*.gz --predictions_path=./predictions/predictions.json`

	Metrics will be displayed to the terminal.
	
## Notes to the solution

### What has been done and why

#### A short intro
1.  I've never worked with QA problems before, so it was a long way from getting familiar with the problem and existing approaches to implementing something that works and can be easily expanded further. I started by reading through [Google's NQ paper](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/1f7b46b5378d757553d3e92ead36bda2e4254244.pdf) itself. The first [strong baseline](https://arxiv.org/pdf/1901.08634.pdf) caught my attention, I started googling more about the approach and discovered a bunch of algorithms on the top of the [leaderboard](https://ai.google.com/research/NaturalQuestions/leaderboard) that follows a similar scheme - [Poolingformer](https://arxiv.org/pdf/2105.04371.pdf), [LSG Transformer](https://huggingface.co/ccdv/lsg-bart-base-4096), [Cluster-Former](https://arxiv.org/pdf/2009.06097v2.pdf) as well as BigBird and Longformer, which are quite popular these days. 

	I've considered these methods to be *"backbone monsters"* -  they leverage large pretrained models, and mostly introduce general new techniques (e.g. new setups of attention), improving SOTA on a bunch of tasks simultaneously. I've decided to check whether there are approaches that leverage the nature of the problem and that are operationally more similar  to how humans think.
    
3.  I discovered three more branches of algorithms during my search:
	* Retrieval-based approaches - [DensePhrases](https://arxiv.org/pdf/2012.12624v3.pdf) and similar ones,
	* Abstractive QA approaches (many of them using retrieval-based knowledge injection) - such as RAG,
	* Finally, I was really happy to discover that someone uses CV-inspired approaches (though I’m not sure how - this is a segmentation paper that’s mentioned on the top of the leaderboard, [D-Former](https://arxiv.org/pdf/2201.00462.pdf)).
	
	Prior to my search, I was sure the most natural methods for extractive QA problems would use exactly the same intuitions as object detection algorithms in CV. My research hasn't changed my opinion. However, these are really time-taking to implement from scratch, and thus I decided to start with a simple yet working *"backbone monster"* approach following JointBert scheme of modeling.  
#### On development
The existing paper that describes JointBert framework in detail was a decent thing to start with given little time left and little computational resources by hand available for code debugging. I followed the approach with several modifications: 
* I used a sliding window approach to generate training samples from original texts. Every training sample was constructed in the following way: `[CLS] question_text [SEP] article_text [SEP]`.
* I changed the data preprocessing scheme by choosing just one sample from a document randomly and taking no extra care of positive/negative class imbalance. 

	The reason for this was that I wouldn't have been able to use all the samples generated a bunch per document due to a lack of computational resources. The same motivation backed up my decision to change the document stride from 128 to 256.
* I haven't used the “[Paragraph=N]”,  “[Table=N]”, and “[List=N]” tokens, introduced in the original paper.
* I kept the number of answer classes the same as mentioned in the original paper.
* I haven't used special tokens, that indicate paragraph numbers, table numbers and etc to simplify development.
* I relied on RoBERTa-Large (not BERT) as a transformer, as it was proven to achieve great results on downstream tasks.
* I used an aggregated NLL loss, a sum of NLL losses for predicted start, end position, and predicted answer class. This loss is equivalent to the loss mentioned in the original paper.
* I used long_answer_candidates to stabilize model predictions by selecting an overlapping span from the candidates.

There have also been some more engineering-related things that affected my solution:
* Due to a lack of computational resources during implementation and debugging, I decided to avoid train set preparation as a separate and independent part of the pipeline, thus preparing train data iteratively, along with training itself. I used a simple extension of pandas `JsonReader` class to read `.jsonl` files iteratively.
* This complicated the usage of HF Trainer (and took a couple of my days to make this decision) , so I had to implement basic functionality myself.

### What would I do if I were to focus on this problem full time for a month

There are quite a lot of things that I'd really want to try. I divide them into the following categories:
1. Data-related 
2. Architecture-related
3. Training-related
4. Postprocessing-related

I want to separately admit, that while there's a huge room for improvement in JointBert approach that I've implemented, there are also other ways to address the QA problem. I mention the things I'd try in the "**Architecture-related**" section.

#### Data-related things
* Sample creation scheme (I believe this is one of the key ingredients for a strong solution):
	* Using Tf-Idf vectorization to choose positive/negative samples that are less similar/more similar to a question respectively.
	* Using a NN to build a probability distribution for choosing negative data entries (this potentially could help more than sparse methods). This can be done with the help of a pretrained model that can score negative samples.
	* Different negative sampling ratios. 
	* Leaving more than 1 sample per document.
* Using longer sequences for training (e.g. with the help of a LongFormer-like model).
* Augmentations:
	* Changing word order.
	* Synonym injection.
	* Changing sentence order.
	* Changing paragraph order.
* Using HTML structural information.
#### Architecture-related:
* Predicting start and end positions conditionally. There are plenty of ways to do this, here are the ones that I'd start with:
	* (inspired by [XLNet](https://arxiv.org/pdf/1906.08237.pdf)) The predicted start token representation is concatenated with all other token representations, after this the resulting vectors are passed to the end token head.
	* The end token head uses attention-like mechanisms to predict the end token index.
* Using Attention over attention layer on the top of the network as described [here](https://arxiv.org/pdf/1909.05286.pdf).
* (inspired by [ELMo](https://arxiv.org/pdf/1802.05365.pdf)) Using intermediate layer outputs averaging to generate the model's output.
* *(not related to the JointBert approach)* Using an object detection approach with a one-stage network. The network (e.g. [YOLO](https://arxiv.org/pdf/2107.08430.pdf)) should act as a span detector, that distinguishes between answer types and backgrounds ([example](https://www.kaggle.com/code/tascj0/a-text-span-detector/notebook)).

#### Training-related
* Analyzing AUC-ROC and AUC-PRC curves during evaluation. This would allow observing how the model's behavior has been changing over time and how its decisions are changing.
* Using explicit AUC PRC maximization / explicit F1 maximization rather than cross-entropy loss minimization. 

	I've worked with [explicit ROC AUC maximization](https://towardsdatascience.com/explicit-auc-maximization-70beef6db14e) once and this approach has a huge potential, despite it's difficult to deal with. Due to the way JointBert framework predicts spans, the problem becomes even more challenging - we need to work with postprocessed predictions and scores.

	As you can recollect, a final score is a difference between the scores of an answer span and a “[CLS]” span. We may use the final score during explicit AUC PRC / F1 maximization, but it would be better to introduce a mechanism that will balance the increase of answer span scores and decrease of "[CLS]" span scores during training.

	We'll also need to pay extra attention to how to take into account the method of generating  a final answer span which is used in the JointBert framework (it uses the candidate spans).

	We could start with using [SOAP](https://proceedings.neurips.cc/paper/2021/file/0dd1bc593a91620daecf7723d2235624-Paper.pdf) implemented in [libAUC](https://github.com/Optimization-AI/LibAUC/blob/main/examples/03_Optimizing_AUPRC_Loss_on_Imbalanced_dataset.ipynb) for AUC PRC maximization.

* Pretraining a model on QA datasets (SQuAD, TriviaQA).
* Using different learning rates for different layers of the network.
* Using Deepspeed with larger backbones and batch sizes.
* Using mixed precision.
* Using [SWA](https://arxiv.org/pdf/1803.05407.pdf).
* Working on improving what the model learns from negative samples.


#### Postprocessing-related
* Using neighboring sample scores to make predictions more "smooth" (It may be the case when the model works poorly predicting spans on the edges of a sample).
*   Working out a robust scheme of using answer candidates (usage of a ranking model for candidates, usage of a classifier that predicts whether a candidate contains a long answer or not, other ways to choose an optimal candidate).

### What would I do if I had way more hardware
A lot of things mentioned above do require extra hardware (sometimes much more extra hardware, e.g. for smart negative samples downsampling). I would additionally mention these points:
* Using HF Trainer with data prepared offline would speed up the processes significantly.
* Finetuning the model for more epochs.
* Pretraining using extra QA datasets like SQuAD , TriviaQA, etc.
* Using all the generated samples during training (having class balance in mind)
* Using larger backbones.
* Hyperparameter tuning (num_epochs, learning rate, batch size - all these would be set up according to a full understanding of why the model behaves in this or that way).
