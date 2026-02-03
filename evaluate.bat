@echo off
REM Quick evaluation script for ECG-ReGen on PTB-XL

echo ========================================
echo ECG-ReGen Evaluation Script
echo ========================================

REM Start evaluation
echo Evaluating model on test set...
python main.py --mode evaluate ^
    --train_data data/ptbxl_train_proper.jsonl ^
    --test_data data/ptbxl_test_proper.jsonl ^
    --checkpoint outputs/best_model.pt ^
    --batch_size 32 ^
    --hidden_dim 768 ^
    --num_workers 4 ^
    --output_dir outputs

echo ========================================
echo Evaluation completed!
echo Results saved to outputs/results.json
echo ========================================
pause
