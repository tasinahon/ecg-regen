@echo off
REM Quick training script for ECG-ReGen on PTB-XL

echo ========================================
echo ECG-ReGen Training Script
echo ========================================

REM Create output directory
if not exist "outputs" mkdir outputs

REM Start training
echo Starting training with paper hyperparameters...
python main.py --mode train ^
    --train_data data/ptbxl_train_proper.jsonl ^
    --val_data data/ptbxl_val_proper.jsonl ^
    --batch_size 32 ^
    --epochs 50 ^
    --learning_rate 5e-5 ^
    --weight_decay 0.01 ^
    --mem_mask_ratio 0.75 ^
    --mlm_mask_ratio 0.15 ^
    --hidden_dim 768 ^
    --num_workers 4 ^
    --save_every 10 ^
    --output_dir outputs

echo ========================================
echo Training completed!
echo ========================================
