sudo docker run --rm -it --gpus all -v `pwd`:/app -p 8099:8099 tf15 /bin/bash
cd app

python dataset_tool.py create_from_images_raw datasets/afhq_labeld afhq --res_log2 9

python run_training.py --num-gpus=2 --data-dir=~/datasets --config=config-e \
  --dataset=afhq --mirror-augment=true

python run_training.py --num-gpus=2 --data-dir=datasets --config=config-e   --dataset=afhq --mirror-augment=true --min-h=4 --min-w=4 --res-log2=7
python run_training.py --num-gpus=2 --data-dir=datasets --config=config-e \
  --dataset=afhq --mirror-augment=true --min-h=4 --min-w=4 --res-log2=7 --cond True