import os
from typing import Tuple, Union

import torch
from datasets import Dataset
from torch import nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel

from utils.utils import set_seed


class ClassificationHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int):
        super(ClassificationHead, self).__init__()

        self.out_proj = nn.Linear(hidden_size, num_labels)
        self.dropout = nn.Dropout()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.dropout(features)
        x = self.out_proj(x)
        return x


class PreTrainedAndClassificationHead(nn.Module):
    def __init__(self, pretrained_model_name: str, num_labels: int):
        super(PreTrainedAndClassificationHead, self).__init__()

        self.pretrained_model = AutoModel.from_pretrained(pretrained_model_name)
        model_output_size = self.pretrained_model.encoder.layer[
            -1
        ].output.dense.out_features
        self.classifier = ClassificationHead(
            hidden_size=model_output_size, num_labels=num_labels
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states=True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        output_pretrained_model = self.pretrained_model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )
        pooler_layer = output_pretrained_model.pooler_output
        logits = self.classifier(pooler_layer)

        if output_hidden_states:
            return output_pretrained_model, logits
        else:
            return logits


def train_model(
    model: PreTrainedAndClassificationHead,
    train: Dataset,
    lr: float,
    num_epochs: int,
    train_batch_size: int,
    save_dir: str,
    device: str,
    train_full_model: bool = False,
    random_state: int = 42,
) -> None:
    set_seed(random_state)
    train_dataloader = DataLoader(train, batch_size=train_batch_size, shuffle=True)

    model.to(device)

    if train_full_model:
        # Unfreeze pretrained model's weights
        for param in model.pretrained_model.parameters():
            param.requires_grad = True
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        model.train()

    else:
        #  Freeze pretrained model's weights
        for param in model.pretrained_model.parameters():
            param.requires_grad = False
        optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=lr)
        model.classifier.train()

    for epoch in range(num_epochs):
        total_loss = 0
        total_correct = 0
        total_samples = 0

        with tqdm(
            train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"
        ) as t:
            for batch in t:
                optimizer.zero_grad()

                #  Batch data
                inputs = batch["input_ids"]
                attention_masks = batch["attention_mask"]
                labels = batch["labels"]

                #  Prediction, and optimization
                with autocast():
                    logits = model(
                        inputs,
                        attention_mask=attention_masks,
                        output_hidden_states=False,
                    )
                    loss = nn.CrossEntropyLoss()(logits, labels)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                #  Accuracy
                predictions = torch.argmax(logits, 1)

                total_correct += (predictions == labels).sum().item()
                total_samples += labels.size(
                    0
                )  # Add the batch size to the total samples
                accuracy = total_correct / total_samples

                t.set_postfix(
                    loss=total_loss / len(train_dataloader), accuracy=accuracy
                )

    #  Save classifier's weights
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if train_full_model:
        torch.save(
            model.pretrained_model.state_dict(), f"{save_dir}/pretrained_model.pth"
        )
    torch.save(model.classifier.state_dict(), f"{save_dir}/classification_head.pth")
