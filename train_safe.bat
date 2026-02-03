@echo off
REM Safe training configuration to prevent out-of-memory errors
REM Uses smaller batch size and no data loading workers

python main.py ^
    --mode train ^
    --train_data data/ptbxl_train_proper.jsonl ^
    --val_data data/ptbxl_val_proper.jsonl ^
    --batch_size 8 ^
    --epochs 50 ^
    --learning_rate 5e-5 ^
    --num_workers 0 ^
    --output_dir outputs

pause
