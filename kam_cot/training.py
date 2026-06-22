"""
KAM-CoT Training Pipeline (2-Stage)
====================================
Training 2 giai đoạn (Decoupled 2-Stage):

  Stage 1 — Rationale Generation:
    Huấn luyện model sinh chuỗi suy luận (chain-of-thought rationale)
    từ input đa phương thức.

  Stage 2 — Answer Prediction:
    Đóng băng encoder, fine-tune decoder để dự đoán đáp án
    (có rationale làm ngữ cảnh bổ trợ).

Cả 2 stage dùng chung kiến trúc, huấn luyện độc lập.
"""

import os
import time
import json
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.optim import AdamW
from torch.optim.lr_scheduler import get_linear_schedule_with_warmup


# ============================================================================
#  Cấu hình Training
# ============================================================================

@dataclass
class TrainingConfig:
    """
    Cấu hình training KAM-CoT.
    
    Các tham số có thể thay đổi dễ dàng:
      - batch_size:     giảm nếu OOM trên T4 (mặc định 4)
      - learning_rate:  learning rate (mặc định 3e-5)
      - num_epochs:     số epoch (mặc định 10)
      - stage:          1 hoặc 2
      - freeze_encoder: True cho Stage 2
      - fp16:           True để dùng mixed precision (tiết kiệm VRAM)
    """
    # Data
    max_text_length: int = 512
    max_target_length: int = 128
    max_nodes: int = 200

    # Training
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-5
    warmup_steps: int = 100
    num_epochs: int = 10
    max_steps: int = -1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    dropout: float = 0.1

    # Stage
    stage: int = 1
    freeze_encoder: bool = False

    # Logging & Save
    output_dir: str = "./kam_cot_output"
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 200
    save_total_limit: int = 3

    # Hardware
    fp16: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)


# ============================================================================
#  Dataset
# ============================================================================

class KAMCoTDataset(Dataset):
    """
    Dataset cho KAM-CoT.

    Mỗi item là dict với keys:
      - question: str (bắt buộc)
      - context: str (tùy chọn)
      - choices: List[str] (tùy chọn)
      - rationale: str (dùng cho Stage 1)
      - answer: str (dùng cho Stage 2)
      - image_path: str (tùy chọn)
      - pixel_values: Tensor (tùy chọn, nếu đã load sẵn)
    """
    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        image_processor=None,
        stage: int = 1,
        kg_extractor=None,
        node_embed_fn=None,
        max_text_length: int = 512,
        max_target_length: int = 128,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.stage = stage
        self.kg_extractor = kg_extractor
        self.node_embed_fn = node_embed_fn
        self.max_text_length = max_text_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.data)

    def _get_text(self, item: Dict) -> str:
        """Xây dựng input text từ question, context, choices."""
        parts = []
        if item.get('context'):
            parts.append(f"Context: {item['context']}")
        parts.append(f"Question: {item.get('question', '')}")
        if item.get('choices'):
            choices = ' '.join(
                f"({chr(65+i)}) {c}" for i, c in enumerate(item['choices'])
            )
            parts.append(f"Choices: {choices}")
        return ' '.join(parts)

    def _get_target(self, item: Dict) -> str:
        """Xây dựng target text theo stage."""
        if self.stage == 1:
            return item.get('rationale', '')
        else:
            rationale = item.get('rationale', '')
            answer = item.get('answer', '')
            if rationale:
                return f"{rationale} Therefore, the answer is {answer}."
            return f"The answer is {answer}."

    def _load_image(self, item: Dict):
        if 'pixel_values' in item and item['pixel_values'] is not None:
            return item['pixel_values']
        if not item.get('image_path'):
            return None
        try:
            from PIL import Image
            img = Image.open(item['image_path']).convert('RGB')
            if self.image_processor is not None:
                return self.image_processor(img, return_tensors='pt')['pixel_values'].squeeze(0)
        except Exception as e:
            print(f"[WARN] Load image failed: {e}")
        return None

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        text = self._get_text(item)
        target = self._get_target(item)

        # Tokenize
        inputs = self.tokenizer(
            text, max_length=self.max_text_length,
            truncation=True, padding='max_length', return_tensors='pt',
        )
        targets = self.tokenizer(
            target, max_length=self.max_target_length,
            truncation=True, padding='max_length', return_tensors='pt',
        )

        result = {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': targets['input_ids'].squeeze(0),
        }
        # -100 = ignore index trong CrossEntropyLoss
        result['labels'][result['labels'] == self.tokenizer.pad_token_id] = -100

        # Image
        pixel_vals = self._load_image(item)
        if pixel_vals is not None:
            result['pixel_values'] = pixel_vals

        # Knowledge Graph
        if self.kg_extractor is not None and self.node_embed_fn is not None:
            kg = self.kg_extractor.build_graph_data(text, self.node_embed_fn)
            result['kg_node_features'] = kg['node_features']
            result['kg_edge_index'] = kg['edge_index']
            result['kg_edge_type'] = kg['edge_type']

        return result


def kam_cot_collate(batch: List[Dict]) -> Dict:
    """
    Collate function xử lý variable-size tensors (KG edges).
    """
    result = {}

    # Các key tensor thông thường
    for key in ('input_ids', 'attention_mask', 'labels'):
        result[key] = torch.stack([b[key] for b in batch])

    # pixel_values
    if 'pixel_values' in batch[0]:
        imgs = [b['pixel_values'] for b in batch if 'pixel_values' in b]
        if imgs:
            result['pixel_values'] = torch.stack(imgs)
        else:
            result['pixel_values'] = None

    # KG data
    if 'kg_node_features' in batch[0]:
        # Xác định p (số nodes mỗi sample)
        p_vals = [b['kg_node_features'].size(0) for b in batch if 'kg_node_features' in b]
        if p_vals:
            p = p_vals[0]  # tất cả đều = max_nodes

            # node_features: (batch * p, d)
            result['kg_node_features'] = torch.cat(
                [b['kg_node_features'] for b in batch if 'kg_node_features' in b], dim=0
            )

            # edge_index: cộng offset theo số nodes mỗi sample
            edges = [b['kg_edge_index'] for b in batch if 'kg_edge_index' in b]
            batched_edges = [
                e + i * p for i, e in enumerate(edges)
            ]
            result['kg_edge_index'] = torch.cat(batched_edges, dim=1)

            # edge_type: concat
            result['kg_edge_type'] = torch.cat(
                [b['kg_edge_type'] for b in batch if 'kg_edge_type' in b], dim=0
            )

            # kg_batch: (batch * p,) — chi sample nào thuộc node nào
            result['kg_batch'] = torch.repeat_interleave(
                torch.arange(len(batch)),
                p
            )

    return result


# ============================================================================
#  Trainer
# ============================================================================

class KAMCoTTrainer:
    """
    Trainer cho KAM-CoT.
    
    Hỗ trợ:
      - Mixed precision (FP16) — tiết kiệm VRAM trên T4
      - Gradient accumulation — batch size ảo lớn hơn
      - Linear schedule với warmup
      - 2-stage training
    """
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        tokenizer,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
    ):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.device = torch.device(config.device)
        self.model.to(self.device)

        # Optimizer
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        params = [
            {'params': [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': config.weight_decay},
            {'params': [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': 0.0},
        ]
        self.optimizer = AdamW(params, lr=config.learning_rate, eps=1e-8)

        # DataLoader
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=kam_cot_collate,
            num_workers=0,
        )
        self.eval_loader = None
        if eval_dataset is not None:
            self.eval_loader = DataLoader(
                eval_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                collate_fn=kam_cot_collate,
                num_workers=0,
            )

        # Scheduler
        total = len(self.train_loader) * config.num_epochs
        if config.max_steps > 0:
            total = config.max_steps
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=config.warmup_steps,
            num_training_steps=total,
        )

        # FP16 scaler
        self.use_fp16 = config.fp16 and config.device == 'cuda'
        self.scaler = torch.cuda.amp.GradScaler() if self.use_fp16 else None

        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        self.tr_loss = 0.0

        print(f"\n[KAM-CoT Trainer] Stage {config.stage}")
        print(f"  Device: {config.device}  Batch: {config.batch_size}")
        print(f"  Accumulation: {config.gradient_accumulation_steps}  FP16: {config.fp16}")
        print(f"  Max nodes: {config.max_nodes}  Max text len: {config.max_text_length}")

    def _move_batch(self, batch: Dict) -> Dict:
        """Chuyển batch sang device."""
        on_device = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                on_device[k] = v.to(self.device)
            else:
                on_device[k] = v
        return on_device

    def train_step(self, batch: Dict) -> float:
        """Một step training."""
        batch = self._move_batch(batch)

        with torch.cuda.amp.autocast(enabled=self.use_fp16):
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                pixel_values=batch.get('pixel_values'),
                kg_node_features=batch.get('kg_node_features'),
                kg_edge_index=batch.get('kg_edge_index'),
                kg_edge_type=batch.get('kg_edge_type'),
                labels=batch['labels'],
            )
            loss = outputs['loss']
            loss = loss / self.config.gradient_accumulation_steps

        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss.item() * self.config.gradient_accumulation_steps

    def optimizer_step(self):
        """Clip gradients và step optimizer."""
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)

        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                       self.config.max_grad_norm)

        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.scheduler.step()
        self.optimizer.zero_grad()

    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate trên eval set."""
        if self.eval_loader is None:
            return float('inf')

        self.model.eval()
        total_loss = 0.0
        n = 0

        for batch in self.eval_loader:
            batch = self._move_batch(batch)
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                pixel_values=batch.get('pixel_values'),
                kg_node_features=batch.get('kg_node_features'),
                kg_edge_index=batch.get('kg_edge_index'),
                kg_edge_type=batch.get('kg_edge_type'),
                labels=batch['labels'],
            )
            total_loss += outputs['loss'].item()
            n += 1

        self.model.train()
        return total_loss / max(n, 1)

    def save(self, name: str = "checkpoint"):
        """Save model checkpoint."""
        path = os.path.join(self.config.output_dir, name)
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path, 'model.pt'))
        self.tokenizer.save_pretrained(path)
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(vars(self.config), f, indent=2, default=str)
        print(f"[SAVE] {path}")

    def train(self):
        """Training loop chính."""
        print(f"\n{'='*50}")
        print(f"  Training Stage {self.config.stage}")
        print(f"{'='*50}\n")

        self.model.train()
        self.model.zero_grad()
        start = time.time()

        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            epoch_loss = 0.0
            epoch_steps = 0

            for step, batch in enumerate(self.train_loader):
                loss = self.train_step(batch)
                epoch_loss += loss
                epoch_steps += 1
                self.global_step += 1
                self.tr_loss += loss

                if self.global_step % self.config.gradient_accumulation_steps == 0:
                    self.optimizer_step()

                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    avg = self.tr_loss / self.config.logging_steps
                    lr = self.scheduler.get_last_lr()[0]
                    elapsed = time.time() - start
                    print(
                        f"  Stage {self.config.stage} | Epoch {epoch+1} | "
                        f"Step {self.global_step} | Loss {avg:.4f} | "
                        f"LR {lr:.2e} | {elapsed:.0f}s"
                    )
                    self.tr_loss = 0.0

                # Evaluation
                if self.global_step % self.config.eval_steps == 0 and self.eval_loader is not None:
                    eval_loss = self.evaluate()
                    print(f"  >> Eval loss: {eval_loss:.4f}")
                    if eval_loss < self.best_loss:
                        self.best_loss = eval_loss
                        self.save("best")
                    self.model.train()

                # Save
                if self.global_step % self.config.save_steps == 0:
                    self.save(f"step-{self.global_step}")

                if 0 < self.config.max_steps <= self.global_step:
                    break

            # End epoch
            avg = epoch_loss / max(epoch_steps, 1)
            print(f"  [Epoch {epoch+1}] Avg loss: {avg:.4f}\n")

            # Flush gradient
            if step % self.config.gradient_accumulation_steps != 0:
                self.optimizer_step()

            if 0 < self.config.max_steps <= self.global_step:
                break

        # Final save
        self.save(f"stage{self.config.stage}_final")
        total = time.time() - start
        print(f"\n{'='*50}")
        print(f"  Complete! Time: {total:.0f}s  Best loss: {self.best_loss:.4f}")
        print(f"{'='*50}\n")

        return {'steps': self.global_step, 'best_loss': self.best_loss}
