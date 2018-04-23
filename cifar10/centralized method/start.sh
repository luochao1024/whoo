python3 cnn_synchronous_method.py --job_name ps --task_index 0 & #>/dev/null &
python3 cnn_synchronous_method.py --job_name worker --task_index 0 &
python3 cnn_synchronous_method.py --job_name worker --task_index 1 &
python3 cnn_synchronous_method.py --job_name worker --task_index 2 &
python3 cnn_synchronous_method.py --job_name worker --task_index 3
