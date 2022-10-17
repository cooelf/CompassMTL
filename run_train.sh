python run_compass_mtl.py \
 --model_type all --mlm_probability 0.25 --mlm_weight 0.1 --task=all --maximum 10000 \
 --model_name_or_path microsoft/deberta-v3-large --report_to none --data_dir="data/corpus" \
 --do_train --learning_rate 6e-6 --num_train_epochs 6 --warmup_steps 500 --weight_decay 0.1 --max_seq_length 512 \
 --per_gpu_eval_batch_size=4 --per_device_train_batch_size=4 \
 --output_dir compass_mtl --save_strategy="epoch" --overwrite_output --gradient_accumulation_steps 2