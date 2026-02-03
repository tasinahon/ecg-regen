# Understanding Report Quality Beyond Loss

## Why Loss Alone Isn't Enough

You're right to be concerned! Here's why:

1. **Loss measures training objective, not report quality**
   - MEM loss: ECG feature reconstruction
   - MLM loss: Masked word prediction  
   - ETM loss: ECG-text matching
   - None directly measure "is the report good?"

2. **Loss can decrease while reports stay bad**
   - Model learns patterns but not clinical reasoning
   - Overfits to common phrases
   - Matches style but not content

## What to Monitor Instead

### 1. Automatic Metrics (Run `monitor_training.py`)

```bash
python monitor_training.py --checkpoint outputs/best_model.pt --num_samples 10
```

**BLEU Scores (0-1, higher is better)**
- BLEU-1 (0.30+): Individual medical terms correct
- BLEU-2 (0.25+): Word pairs match
- BLEU-4 (0.15+): Sentence structure similar

**ROUGE-L (0-1, higher is better)**
- 0.40+: Good overlap with reference
- Measures longest common subsequence

**METEOR (0-1, higher is better)**  
- 0.30+: Considers synonyms and stemming
- Better for medical text than BLEU

### 2. Clinical Evaluation (Manual Check)

Look at generated reports and ask:

✓ **Correctness**: Are diagnoses accurate?
✓ **Completeness**: All key findings mentioned?
✓ **Coherence**: Readable and logical?
✓ **Specificity**: Detailed vs generic?

### 3. Red Flags

⚠️ **Repetitive text**: "normal ekg normal ekg normal..."
⚠️ **Generic reports**: Always says "normal" or "abnormal"  
⚠️ **Nonsensical**: "sinus atrial fibrillation" (contradiction)
⚠️ **Hallucinations**: Mentions findings not in ECG

## Training Progress Stages

**Early (Epochs 1-10)**
- Loss: High and decreasing fast
- Reports: Random or generic ("normal ekg")
- Metrics: BLEU < 0.10

**Mid (Epochs 10-30)**  
- Loss: Moderate and steady decrease
- Reports: Start using medical terms correctly
- Metrics: BLEU 0.15-0.25

**Late (Epochs 30-50)**
- Loss: Low and plateauing
- Reports: Should be clinically meaningful
- Metrics: BLEU 0.25-0.35 (good for medical reports!)

## Expected Final Performance

From the ECG-ReGen paper:
- **BLEU-1**: ~0.35
- **BLEU-4**: ~0.18
- **ROUGE-L**: ~0.45

Medical reports have LOWER scores than natural images because:
- Multiple valid phrasings for same finding
- Technical terminology variations
- Different radiologists emphasize different aspects

## How to Use This

1. **Every 5 epochs**: Check validation samples printed during training
2. **Every 10 epochs**: Run `monitor_training.py` to see actual generated reports
3. **After training**: Full evaluation with `evaluation.py`

4. **Key question**: "Would a doctor find this report useful?"
   Not: "Does it exactly match the reference?"
