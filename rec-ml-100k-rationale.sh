CUDA_VISIBLE_DEVICES=0 python main.py --dataset ml100k  --prompt_format REC-P --bs 8 --eval_bs 8 --epoch 10 --use_generate --output_dir experiments --input_len 512 --output_len 256 --stage 1

python main.py --dataset ml100k  --prompt_format REC-P --bs 8 --eval_bs 8 --epoch 3 --use_generate --output_dir experiments --input_len 128 --output_len 192 --stage 1 --sample_ratio 0.1
