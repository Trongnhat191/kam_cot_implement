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

app = modal.App("ScienceQA-KAM-CoT-Evaluation")


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
    """
    processed = []
    for item in raw_dataset:
        answer_idx = item.get('answer', 0)
        choices = item.get('choices', [])
        answer_text = choices[answer_idx] if answer_idx < len(choices) else str(answer_idx)

        solution_text = item.get('solution', '').strip()
        if not solution_text:
            continue  # skip samples without rationale

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
def evaluate():
    os.chdir("/root/ScienceQA_Project")

    print("=" * 60)
    print("  KAM-CoT Evaluation — ScienceQA Test Split")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load model config & tokenizer
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

    print("\n[1] Loading tokenizer & image processor...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_CONFIG['model_name'])
    image_processor = DetrImageProcessor.from_pretrained(MODEL_CONFIG['vision_model'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"    Device: {device}")

    # ------------------------------------------------------------------
    # 2. Load test data
    # ------------------------------------------------------------------
    from datasets import load_dataset

    print("\n[2] Loading ScienceQA test split...")
    ds_test = load_dataset(
        "derek-thomas/ScienceQA",
        split="test",
        download_mode="force_redownload",
    )
    print(f"    Test samples: {len(ds_test)}")
    test_data = _prepare_scienceqa_data(ds_test, image_processor)
    print(f"    Test samples (after filter): {len(test_data)}")

    # ------------------------------------------------------------------
    # 3. Setup KG extractor
    # ------------------------------------------------------------------
    from kam_cot import ConceptNetExtractor

    print("\n[3] Setting up KG extractor...")
    kg_extractor = ConceptNetExtractor(
        language='en',
        max_nodes=50,
        max_hops=2,
        db_path="/data/conceptnet.db",
    )

    KG_CACHE_DIR = "/output/kg_cache"

    # ------------------------------------------------------------------
    # 4. Load Stage 1 model for rationale generation
    # ------------------------------------------------------------------
    print("\n[4] Loading Stage 1 model for rationale generation...")
    model_s1 = KAMCoTModel(**MODEL_CONFIG).to(device)
    node_embed_fn_s1 = model_s1.get_node_embed_fn(tokenizer)

    # Try to load best model first, then fallback to final
    best_s1_path = "/output/kam_cot_output/best/model.pt"
    final_s1_path = "/output/kam_cot_output/stage1_final/model.pt"
    
    if os.path.exists(best_s1_path):
        s1_load_path = best_s1_path
        print(f"    Loading best Stage 1 model from {s1_load_path}")
    elif os.path.exists(final_s1_path):
        s1_load_path = final_s1_path
        print(f"    Loading final Stage 1 model from {s1_load_path}")
    else:
        print(f"    ERROR: No Stage 1 model found!")
        print(f"    Checked paths:")
        print(f"      - {best_s1_path}")
        print(f"      - {final_s1_path}")
        return

    model_s1.load_state_dict(torch.load(s1_load_path, map_location=device))
    model_s1.eval()
    print(f"    Stage 1 model loaded successfully")

    # ------------------------------------------------------------------
    # 5. Evaluate Stage 1 — Generate rationales & compute ROUGE-L
    # ------------------------------------------------------------------
    from kam_cot.training import KAMCoTDataset, kam_cot_collate, TrainingConfig
    from torch.utils.data import DataLoader
    from rouge_score import rouge_scorer
    import copy

    print("\n[5] Evaluating Stage 1 — Rationale Generation (ROUGE-L)...")

    train_config = TrainingConfig(
        stage=1,
        batch_size=16,
        max_text_length=512,
        max_target_length=512,
        max_nodes=50,
    )

    test_dataset_s1 = KAMCoTDataset(
        data=test_data,
        tokenizer=tokenizer,
        image_processor=image_processor,
        stage=1,
        kg_extractor=kg_extractor,
        node_embed_fn=node_embed_fn_s1,
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

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = []
    generated_rationales = []
    sample_idx = 0

    print(f"    Generating rationales for {len(test_data)} samples...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader_s1):
            if batch_idx % 10 == 0:
                print(f"      Processing batch {batch_idx}/{len(test_loader_s1)}...")
            
            batch_on_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                               for k, v in batch.items()}
            gen_ids = model_s1.generate(
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
    print(f"\n    ✓ Stage 1 ROUGE-L: {avg_rougeL:.4f} ({len(rouge_scores)} samples)")

    # ------------------------------------------------------------------
    # 6. Load Stage 2 model for answer prediction
    # ------------------------------------------------------------------
    print("\n[6] Loading Stage 2 model for answer prediction...")
    model_s2 = KAMCoTModel(**MODEL_CONFIG).to(device)
    node_embed_fn_s2 = model_s2.get_node_embed_fn(tokenizer)

    best_s2_path = "/output/kam_cot_output_stage2/best/model.pt"
    final_s2_path = "/output/kam_cot_output_stage2/stage2_final/model.pt"
    
    if os.path.exists(best_s2_path):
        s2_load_path = best_s2_path
        print(f"    Loading best Stage 2 model from {s2_load_path}")
    elif os.path.exists(final_s2_path):
        s2_load_path = final_s2_path
        print(f"    Loading final Stage 2 model from {s2_load_path}")
    else:
        print(f"    ERROR: No Stage 2 model found!")
        print(f"    Checked paths:")
        print(f"      - {best_s2_path}")
        print(f"      - {final_s2_path}")
        return

    model_s2.load_state_dict(torch.load(s2_load_path, map_location=device))
    model_s2.eval()
    print(f"    Stage 2 model loaded successfully")

    # ------------------------------------------------------------------
    # 7. Evaluate Stage 2 — End-to-End Answer Prediction
    # ------------------------------------------------------------------
    print("\n[7] Evaluating Stage 2 — End-to-End Answer Prediction (Accuracy)...")
    print("    Using generated rationales from Stage 1 as input (not ground truth)")

    train_config_s2 = TrainingConfig(
        stage=2,
        batch_size=16s,
        max_text_length=512,
        max_target_length=32,
        max_nodes=50,
    )

    # Replace ground truth rationales with generated ones
    test_data_e2e = copy.deepcopy(test_data)
    for i, gen_rat in enumerate(generated_rationales):
        test_data_e2e[i]['rationale'] = gen_rat

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

    correct = 0
    total = 0
    sample_idx = 0

    print(f"    Predicting answers for {len(test_data)} samples...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader_s2):
            if batch_idx % 10 == 0:
                print(f"      Processing batch {batch_idx}/{len(test_loader_s2)}...")
            
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
                if 'the answer is' in pred_text:
                    pred_answer = pred_text.split('the answer is')[-1].strip().rstrip('.')
                else:
                    pred_answer = pred_text
                if pred_answer == ref_answer:
                    correct += 1
                total += 1
                sample_idx += 1

    accuracy = correct / max(total, 1)
    print(f"\n    ✓ Stage 2 Accuracy (E2E): {accuracy:.4f} ({correct}/{total})")

    # ------------------------------------------------------------------
    # 8. Save results
    # ------------------------------------------------------------------
    import json

    print(f"\n{'='*60}")
    print(f"  FINAL TEST RESULTS (End-to-End Pipeline)")
    print(f"  Stage 1 ROUGE-L:         {avg_rougeL:.4f}")
    print(f"  Stage 2 Accuracy (E2E):  {accuracy:.4f} ({correct}/{total})")
    print(f"{'='*60}\n")

    result_path = "/output/kam_cot_output_stage2/test_results_evaluation.json"
    with open(result_path, 'w') as f:
        json.dump({
            'test_stage1_rougeL': avg_rougeL,
            'test_stage2_accuracy_e2e': accuracy,
            'test_stage2_correct': correct,
            'test_stage2_total': total,
            'stage1_model_path': s1_load_path,
            'stage2_model_path': s2_load_path,
        }, f, indent=2)
    print(f"    Results saved to {result_path}")
    output_volume.commit()


@app.local_entrypoint()
def local_main():
    evaluate.remote()
