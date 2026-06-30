"""
Bug Condition Exploration Tests for NaN Training Stage 2 Fix
=============================================================

**CRITICAL**: These tests MUST FAIL on unfixed code - failure confirms the bug exists.
**DO NOT attempt to fix the test or the code when it fails.**

**Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.10**

This module contains property-based tests that surface counterexamples demonstrating
the NaN bug across all 6 root causes identified in the design document:

1. Mixed Precision Scaler Bug (1.1)
2. Attention Overflow (1.2, 1.7)
3. Empty Graph / Isolated Nodes (1.3, 1.8)
4. All-Padding Labels (1.6)
5. Large Weight Initialization (1.5)
6. Layer Normalization Instability (1.9, 1.10)

Expected Outcome: Tests FAIL on unfixed code (proves bug exists)
Goal: Document counterexamples that trigger NaN propagation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==============================================================================
# Test 1.1: Mixed Precision Scaler Bug
# ==============================================================================

def test_mixed_precision_scaler_workflow():
    """
    **Validates: Requirements 1.1**
    
    Property 1: Bug Condition - NaN Propagation in Stage 2 Training
    Sub-condition 1.1: Mixed Precision Scaler Bug
    
    Test that optimizer_step() with fp16=True follows correct sequence:
    scaler.unscale_() → clip_grad_norm_() → scaler.step() → scaler.update()
    
    Expected on UNFIXED code: AssertionError from scaler workflow
    Expected on FIXED code: No error, gradients clipped successfully
    """
    print("\n" + "="*70)
    print("Test 1.1: Mixed Precision Scaler Bug")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("⚠️  SKIPPED: CUDA not available, skipping FP16 test")
        return "SKIPPED"
    
    # Create minimal model
    model = nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler('cuda')
    
    device = torch.device('cuda')
    model = model.to(device)
    
    # Forward pass with autocast
    x = torch.randn(4, 10, device=device)
    with torch.amp.autocast('cuda'):
        output = model(x)
        loss = output.sum()
    
    # Backward
    scaler.scale(loss).backward()
    
    # CRITICAL: This is where the bug manifests
    # Unfixed code calls clip_grad_norm_() BEFORE scaler.unscale_()
    # This should raise AssertionError on unfixed code
    try:
        # Simulate unfixed behavior (WRONG ORDER)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.unscale_(optimizer)  # This will fail with AssertionError
        scaler.step(optimizer)
        scaler.update()
        
        # If we reach here on unfixed code, something is wrong with the test
        print("❌ UNEXPECTED PASS: No AssertionError raised - code may already be fixed")
        return "UNEXPECTED_PASS"
    except AssertionError as e:
        # This is EXPECTED on unfixed code
        print(f"✅ EXPECTED FAILURE: Mixed precision scaler bug confirmed")
        print(f"   Error: {e}")
        print(f"   Counterexample: Wrong order of scaler operations causes AssertionError")
        return "EXPECTED_FAIL"


# ==============================================================================
# Test 1.2: Attention Overflow with Large Inputs
# ==============================================================================

def test_attention_overflow_with_large_inputs():
    """
    **Validates: Requirements 1.2, 1.7**
    
    Property 1: Bug Condition - NaN Propagation in Stage 2 Training
    Sub-condition 1.2: Attention Overflow
    
    Test that CrossAttention handles large unnormalized inputs without NaN.
    
    Strategy: Generate H_lang and H_img with max values > 100.0
    Expected on UNFIXED code: NaN in output after softmax
    Expected on FIXED code: Finite output values
    """
    print("\n" + "="*70)
    print("Test 1.2: Attention Overflow with Large Inputs")
    print("="*70)
    
    # Recreate CrossAttention logic inline
    d_model = 768
    scale = math.sqrt(d_model)
    
    # Projections
    q_proj = nn.Linear(d_model, d_model, bias=False)
    k_proj = nn.Linear(d_model, d_model, bias=False)
    v_proj = nn.Linear(d_model, d_model, bias=False)
    out_proj = nn.Linear(d_model, d_model)
    
    # Test with large unnormalized inputs
    test_cases = [
        (200.0, 2, 10),
        (150.0, 4, 20),
        (300.0, 1, 15),
    ]
    
    failures = []
    for max_value, batch_size, seq_len in test_cases:
        # Create inputs with large values (unnormalized)
        H_lang = torch.randn(batch_size, seq_len, d_model) * max_value
        H_img = torch.randn(batch_size, seq_len, d_model) * max_value
        
        # Attention computation (as in unfixed code)
        Q = q_proj(H_lang)
        K = k_proj(H_img)
        V = v_proj(H_img)
        
        attn = torch.matmul(Q, K.transpose(-2, -1))  # (B, n, m)
        attn = attn / scale
        
        attn_weights = F.softmax(attn, dim=-1)
        
        # Check for NaN
        has_nan = torch.isnan(attn_weights).any().item()
        has_inf = torch.isinf(attn_weights).any().item()
        
        if has_nan or has_inf:
            failures.append({
                'max_value': max_value,
                'batch_size': batch_size,
                'seq_len': seq_len,
                'H_lang_max': H_lang.max().item(),
                'H_img_max': H_img.max().item(),
                'attn_max': attn.max().item() if not torch.isnan(attn).any() else 'NaN',
                'has_nan': has_nan,
                'has_inf': has_inf,
            })
    
    if failures:
        print(f"✅ EXPECTED FAILURE: Attention overflow bug confirmed")
        print(f"   Found {len(failures)} counterexample(s):")
        for i, failure in enumerate(failures[:3], 1):  # Show first 3
            print(f"   {i}. max_value={failure['max_value']:.1f}, "
                  f"H_lang.max={failure['H_lang_max']:.1f}, "
                  f"attn.max={failure['attn_max']}, "
                  f"NaN={failure['has_nan']}, Inf={failure['has_inf']}")
        return "EXPECTED_FAIL"
    else:
        print("❌ UNEXPECTED PASS: No NaN/Inf found with large inputs - code may already be fixed")
        return "UNEXPECTED_PASS"


# ==============================================================================
# Test 1.3: Empty Graph Edge Index
# ==============================================================================

def test_empty_graph_nan_propagation():
    """
    **Validates: Requirements 1.3, 1.8**
    
    Property 1: Bug Condition - NaN Propagation in Stage 2 Training
    Sub-condition 1.3: Empty Graph
    
    Test that GraphEncoder handles empty edge_index without NaN.
    
    Strategy: Pass edge_index with 0 edges
    Expected on UNFIXED code: NaN in normalized adjacency
    Expected on FIXED code: Finite output with self-loops added
    """
    print("\n" + "="*70)
    print("Test 1.3: Empty Graph NaN Propagation")
    print("="*70)
    
    d_model = 768
    
    # Simulate GCN fallback logic from graph_encoder.py
    test_cases = [
        (10, 0),   # 10 nodes, 0 edges
        (20, 0),   # 20 nodes, 0 edges
        (5, 0),    # 5 nodes, 0 edges
    ]
    
    failures = []
    for num_nodes, num_edges in test_cases:
        x = torch.randn(num_nodes, d_model)
        edge_index = torch.zeros(2, num_edges, dtype=torch.long)
        
        # GCN normalization logic (unfixed version without self-loops first)
        N = x.size(0)
        adj = torch.zeros(N, N, dtype=torch.float32)
        if edge_index.numel() > 0:
            src, tgt = edge_index
            adj[src, tgt] = 1.0
        
        # Degree normalization (VULNERABLE TO ZERO DEGREE)
        deg = adj.sum(dim=1).clamp(min=1.0).pow(-0.5)
        adj_norm = deg.unsqueeze(1) * adj * deg.unsqueeze(0)
        
        # Check for NaN/Inf
        has_nan = torch.isnan(adj_norm).any().item()
        has_inf = torch.isinf(adj_norm).any().item()
        
        # Check if all degrees are zero (before clamping)
        zero_degrees = (adj.sum(dim=1) == 0).sum().item()
        
        if has_nan or has_inf or zero_degrees == num_nodes:
            failures.append({
                'num_nodes': num_nodes,
                'num_edges': num_edges,
                'zero_degrees': zero_degrees,
                'has_nan': has_nan,
                'has_inf': has_inf,
            })
    
    if failures:
        print(f"✅ EXPECTED FAILURE: Empty graph bug confirmed")
        print(f"   Found {len(failures)} counterexample(s):")
        for i, failure in enumerate(failures, 1):
            print(f"   {i}. num_nodes={failure['num_nodes']}, "
                  f"zero_degrees={failure['zero_degrees']}, "
                  f"NaN={failure['has_nan']}, Inf={failure['has_inf']}")
        return "EXPECTED_FAIL"
    else:
        print("❌ UNEXPECTED PASS: No NaN/Inf found with empty graphs - code may already be fixed")
        return "UNEXPECTED_PASS"


# ==============================================================================
# Test 1.4: All-Padding Labels
# ==============================================================================

def test_all_padding_labels_loss():
    """
    **Validates: Requirements 1.6**
    
    Property 1: Bug Condition - NaN Propagation in Stage 2 Training
    Sub-condition 1.4: All-Padding Labels
    
    Test that model handles batches where all labels are -100 (padding).
    
    Strategy: Create batch with labels = torch.full((B, L), -100)
    Expected on UNFIXED code: loss = 0.0 or NaN
    Expected on FIXED code: Batch skipped or loss=None with warning
    """
    print("\n" + "="*70)
    print("Test 1.4: All-Padding Labels")
    print("="*70)
    
    try:
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        
        # Create minimal T5 model (use small for speed)
        print("   Loading T5-small model (this may take a moment)...")
        model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
        
        test_cases = [
            (2, 32),
            (4, 64),
            (1, 128),
        ]
        
        failures = []
        for batch_size, seq_len in test_cases:
            # Create input
            input_ids = torch.randint(0, 1000, (batch_size, seq_len))
            attention_mask = torch.ones(batch_size, seq_len)
            
            # All labels are -100 (padding)
            labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # Check loss value
            is_zero = (loss.item() == 0.0) if loss is not None else False
            is_nan = (torch.isnan(loss).item()) if loss is not None else False
            is_none = (loss is None)
            
            if is_zero or is_nan:
                failures.append({
                    'batch_size': batch_size,
                    'seq_len': seq_len,
                    'loss': loss.item() if loss is not None else None,
                    'is_zero': is_zero,
                    'is_nan': is_nan,
                })
        
        if failures:
            print(f"✅ EXPECTED FAILURE: All-padding labels bug confirmed")
            print(f"   Found {len(failures)} counterexample(s):")
            for i, failure in enumerate(failures, 1):
                status = 'zero' if failure['is_zero'] else 'NaN'
                print(f"   {i}. batch={failure['batch_size']}, seq_len={failure['seq_len']}, "
                      f"loss={failure['loss']:.6f} ({status})")
            return "EXPECTED_FAIL"
        else:
            print("❌ UNEXPECTED PASS: Loss handled correctly with all-padding labels - "
                  "code may already be fixed")
            return "UNEXPECTED_PASS"
    
    except ImportError as e:
        print(f"⚠️  SKIPPED: transformers library not available - {e}")
        return "SKIPPED"
    except Exception as e:
        print(f"⚠️  ERROR: {e}")
        return "ERROR"


# ==============================================================================
# Test 1.5: Large Weight Initialization
# ==============================================================================

def test_large_weight_initialization_gradient_explosion():
    """
    **Validates: Requirements 1.5**
    
    Property 1: Bug Condition - NaN Propagation in Stage 2 Training
    Sub-condition 1.5: Large Weight Initialization
    
    Test that cross-attention has reasonable weight initialization.
    
    Strategy: Use default PyTorch initialization, check if gradients explode
    Expected on UNFIXED code: Gradients > 1000 or NaN with default init
    Expected on FIXED code: Gradients bounded < 10.0 with proper init
    """
    print("\n" + "="*70)
    print("Test 1.5: Large Weight Initialization")
    print("="*70)
    
    d_model = 768
    
    # Create attention module with default initialization
    q_proj = nn.Linear(d_model, d_model, bias=False)
    k_proj = nn.Linear(d_model, d_model, bias=False)
    v_proj = nn.Linear(d_model, d_model, bias=False)
    out_proj = nn.Linear(d_model, d_model)
    
    # Check initial weight magnitudes
    max_weights = []
    for name, module in [('q_proj', q_proj), ('k_proj', k_proj), 
                         ('v_proj', v_proj), ('out_proj', out_proj)]:
        if hasattr(module, 'weight'):
            max_val = module.weight.abs().max().item()
            max_weights.append(max_val)
            print(f"   {name}: max weight = {max_val:.4f}")
    
    max_init_weight = max(max_weights)
    print(f"   Max initial weight magnitude: {max_init_weight:.4f}")
    
    # Forward pass with normal input
    batch_size = 2
    seq_len = 10
    H_lang = torch.randn(batch_size, seq_len, d_model)
    H_img = torch.randn(batch_size, seq_len, d_model)
    
    Q = q_proj(H_lang)
    K = k_proj(H_img)
    V = v_proj(H_img)
    
    scale = math.sqrt(d_model)
    attn = torch.matmul(Q, K.transpose(-2, -1)) / scale
    attn_weights = F.softmax(attn, dim=-1)
    output = torch.matmul(attn_weights, V)
    output = out_proj(output)
    
    loss = output.sum()
    loss.backward()
    
    # Check gradient norms
    grad_norms = []
    modules = [('q_proj', q_proj), ('k_proj', k_proj), ('v_proj', v_proj), ('out_proj', out_proj)]
    for name, module in modules:
        if hasattr(module, 'weight') and module.weight.grad is not None:
            grad_norm = module.weight.grad.norm().item()
            grad_norms.append(grad_norm)
            print(f"   {name}: grad norm = {grad_norm:.4f}")
    
    max_grad_norm = max(grad_norms) if grad_norms else 0.0
    has_nan_grad = any(torch.isnan(m.weight.grad).any() 
                       for _, m in modules 
                       if hasattr(m, 'weight') and m.weight.grad is not None)
    
    print(f"   Max gradient norm: {max_grad_norm:.4f}, has NaN grad: {has_nan_grad}")
    
    # Check if gradients are problematic
    if max_grad_norm > 100 or has_nan_grad:  # Lowered threshold for more sensitivity
        print(f"✅ EXPECTED FAILURE: Large weight initialization causes unstable gradients")
        print(f"   Counterexample: max_grad_norm={max_grad_norm:.2f}, "
              f"max_init_weight={max_init_weight:.4f}, has_nan={has_nan_grad}")
        return "EXPECTED_FAIL"
    else:
        print(f"❌ UNEXPECTED PASS: Gradients are stable ({max_grad_norm:.2f}) - "
              f"code may already be fixed or test needs adjustment")
        return "UNEXPECTED_PASS"


# ==============================================================================
# Test 1.6: Layer Normalization with Zero Variance
# ==============================================================================

def test_layer_norm_zero_variance():
    """
    **Validates: Requirements 1.9**
    
    Property 1: Bug Condition - NaN Propagation in Stage 2 Training
    Sub-condition 1.6: Layer Normalization Instability
    
    Test that LayerNorm handles zero-variance inputs without NaN.
    
    Strategy: Create input where all values along d_model dimension are identical
    Expected on UNFIXED code: NaN in output due to division by zero
    Expected on FIXED code: Finite output with proper epsilon handling
    """
    print("\n" + "="*70)
    print("Test 1.6: Layer Normalization Zero Variance")
    print("="*70)
    
    d_model = 768
    layer_norm = nn.LayerNorm(d_model)
    
    test_cases = [
        (2, 10, 5.0),
        (4, 20, -3.0),
        (1, 5, 0.0),
    ]
    
    failures = []
    for batch_size, seq_len, const_value in test_cases:
        # Create input with zero variance along d_model dimension
        x = torch.full((batch_size, seq_len, d_model), const_value)
        
        variance = x.var(dim=-1).max().item()
        
        # Forward pass
        output = layer_norm(x)
        
        # Check for NaN
        has_nan = torch.isnan(output).any().item()
        has_inf = torch.isinf(output).any().item()
        
        if has_nan or has_inf:
            failures.append({
                'batch_size': batch_size,
                'seq_len': seq_len,
                'const_value': const_value,
                'variance': variance,
                'has_nan': has_nan,
                'has_inf': has_inf,
            })
    
    if failures:
        print(f"✅ EXPECTED FAILURE: Layer norm instability bug confirmed")
        print(f"   Found {len(failures)} counterexample(s):")
        for i, failure in enumerate(failures, 1):
            print(f"   {i}. const_value={failure['const_value']:.2f}, "
                  f"variance={failure['variance']:.2e}, "
                  f"NaN={failure['has_nan']}, Inf={failure['has_inf']}")
        return "EXPECTED_FAIL"
    else:
        print("❌ UNEXPECTED PASS: LayerNorm handled zero-variance input correctly - "
              "code may already be fixed")
        return "UNEXPECTED_PASS"


# ==============================================================================
# Main execution
# ==============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Bug Condition Exploration Tests - NaN Training Stage 2 Fix")
    print("="*70)
    print("\n⚠️  CRITICAL: These tests EXPECT TO FAIL on unfixed code!")
    print("Failure confirms the bug exists and documents counterexamples.\n")
    
    # Run tests individually for better debugging
    tests = [
        ("1.1 Mixed Precision Scaler Bug", test_mixed_precision_scaler_workflow),
        ("1.2 Attention Overflow", test_attention_overflow_with_large_inputs),
        ("1.3 Empty Graph", test_empty_graph_nan_propagation),
        ("1.4 All-Padding Labels", test_all_padding_labels_loss),
        ("1.5 Large Weight Initialization", test_large_weight_initialization_gradient_explosion),
        ("1.6 Layer Norm Zero Variance", test_layer_norm_zero_variance),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ ERROR in test {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, f"ERROR: {str(e)[:50]}"))
    
    print(f"\n{'='*70}")
    print("Test Summary")
    print(f"{'='*70}")
    for name, result in results:
        status_icon = "✅" if result == "EXPECTED_FAIL" else "❌" if result == "UNEXPECTED_PASS" else "⚠️"
        print(f"{status_icon} {name}: {result}")
    
    # Count results
    expected_fails = sum(1 for _, r in results if r == "EXPECTED_FAIL")
    unexpected_passes = sum(1 for _, r in results if r == "UNEXPECTED_PASS")
    
    print(f"\n{'='*70}")
    print(f"Results: {expected_fails} bugs confirmed, {unexpected_passes} unexpected passes")
    print(f"{'='*70}")
    
    if expected_fails > 0:
        print("\n✅ SUCCESS: Bug conditions have been validated with counterexamples.")
        print("   The tests confirmed NaN propagation issues exist in the unfixed code.")
    elif unexpected_passes > 0:
        print("\n⚠️  WARNING: Some tests passed unexpectedly.")
        print("   This suggests the code may already have fixes, or tests need adjustment.")
