#!/bin/bash

# arg 1 - starting proc number
# arg 2 - ending proc number
# arg 3 - total number of procs

for i in `seq $1 $2`; do
	echo $i " of " $3 " started"
        python doFlaresGaussProcSubset.py $i $3 >& output.p$i &
	pids[${i}]=$!
done

echo "waiting for processes to finish"

for pid in ${pids[*]}; do
	wait $pid
done

echo "all processes finished!"
