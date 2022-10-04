import torch.nn as nn
import torch

from typing import Tuple


class NQModel(nn.Module):
    """
    Transformer with simple QA and CLS heads. By design, it just makes use of default QA head
    from HF transformers as well as default CLS head from HF transformers
    """

    def __init__(self, model, qa_head, pooler=None, num_labels: int = 3):
        super(NQModel, self).__init__()
        self.transformer = model

        self.pooler = pooler
        self.dropout = nn.Dropout(0.1)

        self.qa_head = qa_head
        self.classifier = nn.Linear(pooler.dense.in_features, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor,
                                                                                      torch.Tensor, torch.Tensor]:

        outputs = self.transformer(input_ids, attention_mask).last_hidden_state

        # obtaining answer start/end scores

        qa_scores = self.qa_head(outputs)
        start_scores, end_scores = qa_scores.split(1, dim=-1)
        start_logits = start_scores.squeeze(-1)
        end_logits = end_scores.squeeze(-1)

        # obtaining classification scores

        pooled_outputs = outputs
        if self.pooler:
            pooled_outputs = self.pooler(outputs)

        pooled_outputs = self.dropout(pooled_outputs)
        classifier_scores = self.classifier(pooled_outputs)

        return start_logits, end_logits, classifier_scores
