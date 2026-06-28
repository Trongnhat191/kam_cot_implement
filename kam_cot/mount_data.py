import modal
import os

# Volumes
data_volume = modal.Volume.from_name("hf-datasets-cache", create_if_missing=True)
cn_volume = modal.Volume.from_name("conceptnet-db", create_if_missing=True)
output_volume = modal.Volume.from_name("kam-cot-output", create_if_missing=True)

image = (
    modal.Image.from_registry("python:3.12-slim")
    .apt_install("curl", "wget", "git", "unzip")
    .uv_pip_install([
        "torch",
        "transformers",
        "datasets",
        "huggingface_hub",
        "einops",
        "pandas",
        "tqdm",
        "timm",
        "conceptnet-lite",
        "Pillow",
    ])
    .env({"HF_DATASETS_CACHE": "/cache/huggingface"})
    .add_local_dir("./", "/root/ScienceQA_Project", copy=True, ignore=['.venv', '__pycache__', 'output_test', '*.egg-info', '.git'])
    .run_commands("cd /root/ScienceQA_Project && pip install -e .")
)

app = modal.App("ScienceQA-KAM-CoT")


from PIL import Image as PILImage


def _resize_and_pad(img: PILImage.Image, size: int = 800) -> PILImage.Image:
    w, h = img.size
    if w >= h:
        new_w, new_h = size, int(h * size / w)
    else:
        new_w, new_h = int(w * size / h), size
    img = img.resize((new_w, new_h), PILImage.BILINEAR)
    result = PILImage.new('RGB', (size, size), (0, 0, 0))
    result.paste(img, ((size - new_w) // 2, (size - new_h) // 2))
    return result


def _prepare_scienceqa_data(raw_dataset, image_processor, stage: int = 1):
    """
    Map derek-thomas/ScienceQA fields → KAMCoTDataset format.

    ScienceQA fields:
      image     : PIL Image or None
      question  : str
      choices   : List[str]
      answer    : int (index into choices)
      solution  : str (reasoning/rationale)
      lecture   : str

    KAMCoTDataset expects:
      question, context, choices, answer (text), rationale, image/pixel_values
    """
    processed = []
    for item in raw_dataset:
        answer_idx = item.get('answer', 0)
        choices = item.get('choices', [])
        answer_text = choices[answer_idx] if answer_idx < len(choices) else str(answer_idx)

        solution_text = item.get('solution', '').strip()
        if not solution_text:
            continue  # skip samples without rationale (would produce all -100 labels)

        sample = {
            'question': item.get('question', ''),
            'context': item.get('lecture', ''),
            'choices': choices,
            'answer': answer_text,
            'rationale': solution_text,
        }

        img = item.get('image')
        if img is not None:
            img = _resize_and_pad(img, size=800)
            try:
                sample['pixel_values'] = image_processor(
                    img, return_tensors='pt'
                )['pixel_values'].squeeze(0)
            except Exception:
                sample['pixel_values'] = None
        else:
            sample['pixel_values'] = None

        processed.append(sample)

    return processed


@app.function(
    image=image,
    gpu="A100",
    volumes={"/cache": data_volume, "/data": cn_volume, "/output": output_volume},
    timeout=28800,
    num_workers=32
)
def main():
    os.chdir("/root/ScienceQA_Project")

    print("=" * 60)
    print("  KAM-CoT Training — ScienceQA")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load model & tokenizer
    # ------------------------------------------------------------------
    from kam_cot import KAMCoTModel
    from transformers import T5Tokenizer, DetrImageProcessor
    import torch

    MODEL_CONFIG = dict(
        model_name="google/flan-t5-base",
        vision_model="facebook/detr-resnet-101-dc5",
        use_vit=False,
        graph_edge_dim=64,
        graph_num_rels=34,
        graph_num_heads=4,
        dropout=0.1,
    )

    print("\n[1] Loading model & tokenizer...")
    model = KAMCoTModel(**MODEL_CONFIG)
    tokenizer = T5Tokenizer.from_pretrained(MODEL_CONFIG['model_name'])
    image_processor = DetrImageProcessor.from_pretrained(MODEL_CONFIG['vision_model'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"    Device: {device}")
    print(f"    Parameters: {model.count_parameters()['total'] / 1e6:.1f}M")

    # ------------------------------------------------------------------
    # 2. Load ScienceQA dataset from cached volume
    # ------------------------------------------------------------------
    from datasets import load_dataset

    print("\n[2] Loading ScienceQA from cached volume...")
    ds = load_dataset(
        "derek-thomas/ScienceQA",
        split="train",
        download_mode="force_redownload",
    )
    # get 20% of the dataset for faster testing
    ds = ds.shuffle(seed=42).select(range(int(len(ds) * 0.4)))
    print(f"    Train samples: {len(ds)}")

    # ------------------------------------------------------------------
    # 3. KG Extractor
    # ------------------------------------------------------------------
    from kam_cot import ConceptNetExtractor

    print("\n[3] Setting up KG extractor...")
    kg_extractor = ConceptNetExtractor(
        language='en',
        max_nodes=50,
        max_hops=2,
        db_path="/data/conceptnet.db",
    )

    node_embed_fn = model.get_node_embed_fn(tokenizer)

    # ------------------------------------------------------------------
    # 4. Prepare dataset
    # ------------------------------------------------------------------
    print("\n[4] Preparing dataset...")
    processed = _prepare_scienceqa_data(ds, image_processor)

    train_data = processed
    eval_data = processed[:100]

    print(f"    Train: {len(train_data)}")
    print(f"    Eval:  {len(eval_data)}")

    sample = train_data[0]
    print(f"    Sample — Q: {sample['question'][:80]}...")
    print(f"    Sample — Rationale: {sample['rationale'][:80]}...")
    print(f"    Sample — Answer: {sample['answer']}")
    print(f"    Sample — has image: {sample['pixel_values'] is not None}")

    # ------------------------------------------------------------------
    # 5. Training config
    # ------------------------------------------------------------------
    from kam_cot.training import TrainingConfig, setup_stage1_training

    train_config = TrainingConfig(
        stage=1,
        freeze_encoder=False,
        batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        warmup_steps=100,
        num_epochs=20,
        max_text_length=256,
        max_target_length=96,
        max_nodes=50,
        fp16=False,
        logging_steps=20,
        eval_steps=500,
        save_steps=1000,
        output_dir="/output/kam_cot_output",
        weight_decay=0.01,
        max_grad_norm=1.0,
    )

    print("\n[5] Training config:")
    print(f"    batch_size={train_config.batch_size}")
    print(f"    accumulation={train_config.gradient_accumulation_steps}")
    print(f"    effective_batch={train_config.batch_size * train_config.gradient_accumulation_steps}")
    print(f"    epochs={train_config.num_epochs}")
    print(f"    lr={train_config.learning_rate}")
    print(f"    max_nodes={train_config.max_nodes}")

    # ------------------------------------------------------------------
    # 6. Train Stage 1
    # ------------------------------------------------------------------
    print("\n[6] Starting Stage 1 training...")
    trainer = setup_stage1_training(
        model=model,
        tokenizer=tokenizer,
        train_data=train_data,
        eval_data=eval_data,
        config=train_config,
        kg_extractor=kg_extractor,
        image_processor=image_processor,
    )

    results = trainer.train()
    output_volume.commit()  # flush writes to persistent storage
    print(f"\n[DONE] Stage 1 complete — {results}")


@app.local_entrypoint()
def local_main():
    main.remote()