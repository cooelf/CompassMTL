python run_compass_mtl.py \
 --eval_task=anli --model_name_or_path compass_mtl --report_to none --task=all --maximum 1000000000  --data_dir="data/corpus" \
 --do_train --do_eval --learning_rate 6e-6 --num_train_epochs 6 --warmup_steps 500 --weight_decay 0.1 --max_seq_length 512 \
 --per_gpu_eval_batch_size=4 --per_device_train_batch_size=2 --model_type mtl \
 --output_dir experiments/anli --overwrite_output --save_strategy "epoch" --evaluation_strategy="epoch" --load_best_model_at_end=True --metric_for_best_model=accuracy