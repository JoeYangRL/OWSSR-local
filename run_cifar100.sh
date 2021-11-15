## CIFAR100 B45-1 steps
# task 1
#python3 main.py --base_task_cls 45 --steps 1 --world_size 4 --num_workers 0 --port 29502 --output_path ./output/cifar100_3 --now_step 1 --lr 0.1 --eval_step 256 --epoch 128 --warmup_epoch 0 --start_fix 20 --timestamp 20211108113629
#python3 main.py --base_task_cls 45 --steps 1 --world_size 4 --num_workers 0 --port 29500 --output_path ./output/cifar100 --now_step 1 --lr 0.1 --eval_step 256 --epoch 128 --warmup_epoch 0 --lambda_oem 0.1 --lambda_socr 1.0 --batch_size 64 --start_fix 20 --timestamp 20211109151438
## CIFAR100 B55-1 steps
# task 1
#python3 main.py --base_task_cls 55 --steps 1 --world_size 4 --num_workers 0 --port 29555 --output_path ./output/cifar100_555 --now_step 1 --lr 0.1 --eval_step 256 --epoch 128 --warmup_epoch 0 --lambda_oem 0.1 --lambda_socr 1.0 --batch_size 64 --start_fix 20 --timestamp 20211109163414

## CIFAR100 B0-10 steps
# task 1
#python3 main.py --steps 10 --world_size 4 --num_workers 0 --port 29502 --output_path ./output/cifar100 --now_step 1 --lr 0.01 --eval_step 256 --epoch 128 --warmup_epoch 0 --start_fix 20 --batch_size 32 --timestamp 20211103110524

###windows
## CIFAR100 B55-1 steps
# task 1
python main.py --base_task_cls 55 --steps 1 --world_size 1 --num_workers 0 --port 29555 --output_path ./output/cifar100_55 --now_step 1 --lr 0.03 --eval_step 256 --epoch 128 --warmup_epoch 0 --lambda_oem 0.1 --lambda_socr 1.0 --batch_size 64 --start_fix 20 --timestamp 20211111150822