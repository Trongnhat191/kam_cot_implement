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
        "rouge-score",
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
    timeout=28800
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
    ds_train = load_dataset(
        "derek-thomas/ScienceQA",
        split="train",
        download_mode="force_redownload",
    )
    ds_eval = load_dataset(
        "derek-thomas/ScienceQA",
        split="validation",
        download_mode="force_redownload",
    )
    # get 40% of train for faster testing
    ds_train = ds_train.shuffle(seed=42).select(range(int(len(ds_train) * 0.4)))
    print(f"    Train samples (40%): {len(ds_train)}")
    print(f"    Eval samples (full): {len(ds_eval)}")

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

    print("\n[4] Preparing dataset...")
    train_data = _prepare_scienceqa_data(ds_train, image_processor)
    eval_data  = _prepare_scienceqa_data(ds_eval,  image_processor)

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
    from kam_cot.training import TrainingConfig, setup_stage1_training, setup_stage2_training

    train_config = TrainingConfig(
        stage=1,
        freeze_encoder=False,
        batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        warmup_steps=100,
        num_epochs=20,
        max_text_length=512,
        max_target_length=512,
        max_nodes=50,
        fp16=False,
        logging_steps=200,
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
    KG_CACHE_DIR = "/output/kg_cache"

    # print("\n[6] Starting Stage 1 training...")
    # trainer = setup_stage1_training(
    #     model=model,
    #     tokenizer=tokenizer,
    #     train_data=train_data,
    #     eval_data=eval_data,
    #     config=train_config,
    #     kg_extractor=kg_extractor,
    #     image_processor=image_processor,
    #     kg_cache_dir=KG_CACHE_DIR,
    # )
    # output_volume.commit()  # flush KG cache to persistent storage

    # results = trainer.train()
    # output_volume.commit()  # flush model checkpoints
    # print(f"\n[DONE] Stage 1 complete — {results}")

    # ------------------------------------------------------------------
    # 7. Re-initialize model for Stage 2 (identical initialization per paper)
    # ------------------------------------------------------------------
    print("\n[7] Preparing for Stage 2 (Answer Prediction)...")
    print("    Re-initializing model from scratch (identical init per paper)...")

    # Paper: "trained separately from identical initializations"
    model_s2 = KAMCoTModel(**MODEL_CONFIG).to(device)
    node_embed_fn_s2 = model_s2.get_node_embed_fn(tokenizer)

    # Cấu hình Stage 2
    train_config_s2 = TrainingConfig(
        stage=2,
        freeze_encoder=False,   # Identical init — train toàn bộ như Stage 1
        batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,     # Cùng LR với Stage 1 (identical setup)
        warmup_steps=100,
        num_epochs=20,
        max_text_length=512,
        max_target_length=64,
        max_nodes=50,
        fp16=False,
        logging_steps=200,
        eval_steps=500,
        save_steps=1000,
        output_dir="/output/kam_cot_output_stage2",
        weight_decay=0.01,
        max_grad_norm=1.0,
    )

    # ------------------------------------------------------------------
    # 8. Train Stage 2
    # ------------------------------------------------------------------
    print("\n[8] Starting Stage 2 training...")
    trainer_s2 = setup_stage2_training(
        model=model_s2,
        tokenizer=tokenizer,
        train_data=train_data,
        eval_data=eval_data,
        config=train_config_s2,
        kg_extractor=kg_extractor,
        image_processor=image_processor,
        freeze_encoder=False,
        kg_cache_dir=KG_CACHE_DIR,
    )

    results_s2 = trainer_s2.train()
    output_volume.commit()  # flush model checkpoints cho Stage 2
    print(f"\n[DONE] Stage 2 complete — {results_s2}")

    # ------------------------------------------------------------------
    # 9. Load test data
    # ------------------------------------------------------------------
    print("\n[9] Final evaluation on ScienceQA test split...")
    import json
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from kam_cot.training import KAMCoTDataset, kam_cot_collate
    from rouge_score import rouge_scorer

    ds_test = load_dataset(
        "derek-thomas/ScienceQA",
        split="test",
        download_mode="force_redownload",
    )
    print(f"    Test samples: {len(ds_test)}")
    test_data = _prepare_scienceqa_data(ds_test, image_processor)
    print(f"    Test samples (after filter): {len(test_data)}")

    # ------------------------------------------------------------------
    # 9a. Test Stage 1 — Rationale Generation (ROUGE-L)
    # ------------------------------------------------------------------
    print("\n[9a] Testing Stage 1 — Rationale Generation (ROUGE-L)...")

    # Load best Stage 1 model
    best_s1_path = os.path.join(train_config.output_dir, "best", "model.pt")
    final_s1_path = os.path.join(train_config.output_dir, "stage1_final", "model.pt")
    s1_load_path = best_s1_path if os.path.exists(best_s1_path) else final_s1_path
    print(f"    Loading Stage 1 model from {s1_load_path}")
    model.load_state_dict(torch.load(s1_load_path, map_location=device))
    model.eval()

    # Stage 1 test dataset (input = Q+C+choices, target = rationale)
    test_dataset_s1 = KAMCoTDataset(
        data=test_data,
        tokenizer=tokenizer,
        image_processor=image_processor,
        stage=1,
        kg_extractor=kg_extractor,
        node_embed_fn=node_embed_fn,
        max_text_length=train_config.max_text_length,
        max_target_length=train_config.max_target_length,
        kg_cache_path=os.path.join(KG_CACHE_DIR, "test_kg_s1.pt"),
    )
    output_volume.commit()

    test_loader_s1 = DataLoader(
        test_dataset_s1,
        batch_size=train_config.batch_size,
        shuffle=False,
        collate_fn=kam_cot_collate,
        num_workers=4,
    )

    # Generate rationales, compute ROUGE-L, and save for End-to-End Stage 2
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = []
    generated_rationales = []  # Lưu rationale sinh ra để truyền cho Stage 2
    sample_idx = 0

    with torch.no_grad():
        for batch in test_loader_s1:
            batch_on_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                               for k, v in batch.items()}
            gen_ids = model.generate(
                input_ids=batch_on_device['input_ids'],
                attention_mask=batch_on_device['attention_mask'],
                pixel_values=batch_on_device.get('pixel_values'),
                kg_node_features=batch_on_device.get('kg_node_features'),
                kg_edge_index=batch_on_device.get('kg_edge_index'),
                kg_edge_type=batch_on_device.get('kg_edge_type'),
                max_length=train_config.max_target_length,
                num_beams=4,
            )
            preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            bsz = batch['input_ids'].size(0)
            for i in range(bsz):
                ref = test_data[sample_idx].get('rationale', '')
                score = scorer.score(ref, preds[i])
                rouge_scores.append(score['rougeL'].fmeasure)
                generated_rationales.append(preds[i])
                sample_idx += 1

    avg_rougeL = sum(rouge_scores) / max(len(rouge_scores), 1)
    print(f"    Stage 1 ROUGE-L: {avg_rougeL:.4f} ({len(rouge_scores)} samples)")

    # ------------------------------------------------------------------
    # 9b. Test Stage 2 — End-to-End Answer Prediction (Accuracy)
    #      Dùng rationale SINH RA từ Stage 1, không dùng ground truth
    # ------------------------------------------------------------------
    print("\n[9b] Testing Stage 2 — End-to-End Answer Prediction (Accuracy)...")
    print("     Using generated rationales from Stage 1 as input (not ground truth)")

    # Load best Stage 2 model
    best_s2_path = os.path.join(train_config_s2.output_dir, "best", "model.pt")
    final_s2_path = os.path.join(train_config_s2.output_dir, "stage2_final", "model.pt")
    s2_load_path = best_s2_path if os.path.exists(best_s2_path) else final_s2_path
    print(f"    Loading Stage 2 model from {s2_load_path}")
    model_s2.load_state_dict(torch.load(s2_load_path, map_location=device))
    model_s2.eval()

    # Tạo bản sao test_data, thay rationale chuẩn bằng rationale sinh ra từ Stage 1
    import copy
    test_data_e2e = copy.deepcopy(test_data)
    for i, gen_rat in enumerate(generated_rationales):
        test_data_e2e[i]['rationale'] = gen_rat

    # Stage 2 test dataset (input = Q+C+choices + rationale sinh bởi Stage 1)
    test_dataset_s2 = KAMCoTDataset(
        data=test_data_e2e,
        tokenizer=tokenizer,
        image_processor=image_processor,
        stage=2,
        kg_extractor=kg_extractor,
        node_embed_fn=node_embed_fn_s2,
        max_text_length=train_config_s2.max_text_length,
        max_target_length=train_config_s2.max_target_length,
        kg_cache_path=os.path.join(KG_CACHE_DIR, "test_kg_s2.pt"),
    )
    output_volume.commit()

    test_loader_s2 = DataLoader(
        test_dataset_s2,
        batch_size=train_config_s2.batch_size,
        shuffle=False,
        collate_fn=kam_cot_collate,
        num_workers=4,
    )

    # Generate answers & compute Accuracy
    correct = 0
    total = 0
    sample_idx = 0

    with torch.no_grad():
        for batch in test_loader_s2:
            batch_on_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                               for k, v in batch.items()}
            gen_ids = model_s2.generate(
                input_ids=batch_on_device['input_ids'],
                attention_mask=batch_on_device['attention_mask'],
                pixel_values=batch_on_device.get('pixel_values'),
                kg_node_features=batch_on_device.get('kg_node_features'),
                kg_edge_index=batch_on_device.get('kg_edge_index'),
                kg_edge_type=batch_on_device.get('kg_edge_type'),
                max_length=train_config_s2.max_target_length,
                num_beams=4,
            )
            preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            bsz = batch['input_ids'].size(0)
            for i in range(bsz):
                ref_answer = test_data[sample_idx].get('answer', '').strip().lower()
                pred_text = preds[i].strip().lower()
                # Extract answer from generated text
                # Format: "The answer is {answer}."
                if 'the answer is' in pred_text:
                    pred_answer = pred_text.split('the answer is')[-1].strip().rstrip('.')
                else:
                    pred_answer = pred_text
                if pred_answer == ref_answer:
                    correct += 1
                total += 1
                sample_idx += 1

    accuracy = correct / max(total, 1)
    print(f"    Stage 2 Accuracy (E2E): {accuracy:.4f} ({correct}/{total})")

    # ------------------------------------------------------------------
    # 10. Summary & Save
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  FINAL TEST RESULTS (End-to-End Pipeline)")
    print(f"  Stage 1 ROUGE-L:         {avg_rougeL:.4f}")
    print(f"  Stage 2 Accuracy (E2E):  {accuracy:.4f} ({correct}/{total})")
    print(f"{'='*60}\n")

    result_path = os.path.join(train_config_s2.output_dir, "test_results.json")
    with open(result_path, 'w') as f:
        json.dump({
            'stage1_train': results,
            'stage2_train': results_s2,
            'test_stage1_rougeL': avg_rougeL,
            'test_stage2_accuracy_e2e': accuracy,
            'test_stage2_correct': correct,
            'test_stage2_total': total,
        }, f, indent=2)
    print(f"    Results saved to {result_path}")
    output_volume.commit()


@app.local_entrypoint()
def local_main():
    main.remote()