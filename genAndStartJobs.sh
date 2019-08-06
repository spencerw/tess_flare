#!/bin/bash

# arg 1 - total number of nodes to use

if [ -z $1 ]
then 
    echo "Error: Specify number of nodes to use as command line argument"
    exit 1
fi

NUM_NODES=$1
TASKS_PER_NODE=28
let NUM_TASKS=$NUM_NODES*$TASKS_PER_NODE

for i in $(seq 1 $NUM_NODES)
do
    let P1=($i-1)*$TASKS_PER_NODE
    let P2=($i*$TASKS_PER_NODE)-1
    cp jobfile_template jobfile$i
    sed -i "s/\^i\^/$i/g" jobfile$i
    sed -i "s/\^p1\^/$P1/g" jobfile$i
    sed -i "s/\^p2\^/$P2/g" jobfile$i
    sed -i "s/\^ptot\^/$NUM_TASKS/g" jobfile$i
done

echo $NUM_NODES "jobfiles created"

for i in $(seq 1 $NUM_NODES)
do
    sbatch jobfile$i
done

echo $NUM_NODES "jobfiles started"
