#!/usr/bin/env bash

#tmux_name, ps_host, worker_host, job_name, task_index
function send_keys {
    tmux send-keys -t $1  "python3 trainer.py \
         --ps_hosts=$2  \
         --worker_hosts=$3 \
         --job_name=$4 --task_index=$5" ENTER
}

if [ -d $(which tmux) ]
then
    echo "Please install tmux"
    exit
fi

tmux kill-server

N_PROCESS=8
BASE_PORT=2222

PS_HOST=""
WORKER_HOST=""

for i in $(seq 0 $N_PROCESS);
do
    if [ $i -eq 0 ];
    then
        PS_HOST="localhost:$BASE_PORT"
        echo "PS_HOST=$PS_HOST"
        tmux new-session -d -s ps
        echo "tmux new session started at ps"
    else
        if [ -z $WORKER_HOST ];
        then
            WORKER_HOST="localhost:$(expr $BASE_PORT + $i)"
        else
            WORKER_HOST="$WORKER_HOST,localhost:$(expr $BASE_PORT + $i)"
        fi
        index=$(expr $i - 1)
        tmux new-session -d -s "w$index"
        echo tmux new session started at w$index

   fi
done
echo "WORKER_HOST=$WORKER_HOST"

for i in $(seq 0 $N_PROCESS)
do
#tmux_name, ps_host, worker_host, job_name, task_index
    if [ $i -eq 0 ]; 
    then
        tmux_name=ps
        job_name=ps
        task_index=$i
    else
        tmux_name=w$(expr $i - 1)
        job_name=worker
        task_index=$(expr $i - 1)
    fi
    send_keys $tmux_name $PS_HOST $WORKER_HOST $job_name $task_index 
done

exit
echo exited
    



tmux new-session -d -s ps
tmux new-session -d -s w0
tmux new-session -d -s w1

# ps
tmux send-keys -t ps  "python3 trainer.py \
     --ps_hosts=localhost:2222 \
     --worker_hosts=localhost:2223,localhost:2224 \
     --job_name=ps --task_index=0" ENTER

# worker
tmux send-keys -t w0 "python3 trainer.py \
     --ps_hosts=localhost:2222 \
     --worker_hosts=localhost:2223,localhost:2224 \
     --job_name=worker --task_index=0" ENTER

tmux send-keys -t w1 "python3 trainer.py \
     --ps_hosts=localhost:2222 \
     --worker_hosts=localhost:2223,localhost:2224 \
     --job_name=worker --task_index=1" ENTER

