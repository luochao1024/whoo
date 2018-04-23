python2 flocking_optimizer_dis_test.py --job_name ps --task_index 0 &>/dev/null &
python2 flocking_optimizer_dis_test.py --job_name worker --task_index 0 &>/dev/null &
python2 flocking_optimizer_dis_test.py --job_name worker --task_index 1 &>/dev/null &
python2 flocking_optimizer_dis_test.py --job_name worker --task_index 2 &>/dev/null &
python2 flocking_optimizer_dis_test.py --job_name worker --task_index 3
