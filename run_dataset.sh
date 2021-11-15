# CIFAR100 B0-2steps
#python3 main_dataset.py --steps 2 --expand_labels 
# CIFAR100 B0-5steps
#python3 main_dataset.py --steps 5 --expand_labels 
# CIFAR100 B0-10steps
#python3 main_dataset.py --steps 10 --expand_labels 
# CIFAR100 B50-2steps
#python3 main_dataset.py --base_task_cls 50 --steps 2 --expand_labels 
# CIFAR100 B50-5steps
#python3 main_dataset.py --base_task_cls 50 --steps 5 --expand_labels 
# CIFAR100 B50-10steps
#python3 main_dataset.py --base_task_cls 50 --steps 10 --expand_labels 

#python3 main_dataset.py --base_task_cls 55 --steps 1 --expand_labels --unlabeled_mu 17

#windows
python main_dataset.py --base_task_cls 55 --steps 1 --unlabeled_mu 17 --root "G:\dataset"