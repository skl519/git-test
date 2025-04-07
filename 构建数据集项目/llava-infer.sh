python scripts/vllm_infer.py \
      --model_name_or_path=/root/autodl-tmp/weight/swift/LLaVA-NeXT-Video-7B-hf\
      --adapter_name_or_path=/root/autodl-tmp/saves/llava_next_video\
      --template=llava_next_video\
      --dataset=my_dataset_infer\
      --save_name=llava-infer.jsonl
