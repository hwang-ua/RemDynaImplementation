#!/bin/sh

SCRIPT=$(dirname "$0")'/run.py'
TASKS_PER_FILE=2

# assert command line arguments valid
if [ "$#" -gt "1" ]
    then
        echo 'usage: ./run.sh [RESULTS_DIR]'
        exit
    fi

# get folder name for results
if [ "$#" == "1" ]
    then
        RESULTS_DIR=$1
    else
        RESULTS_DIR=$(date +%Y-%m-%dT%H:%M:%S%z)
    fi

# collect all tasks
rm tasks_*.sh 2>/dev/null
rm tasks.sh 2>/dev/null
for ALPHA in '0.001' '0.01' '0.1'; do
    for LAMDA in '0.0' '0.9' '1.0'; do
        DIR="$RESULTS_DIR"'/'"$ALPHA"'/'"$LAMDA"
        mkdir -p "$DIR" 2>/dev/null
        ARGS=('--dir '"$DIR"
              '--alpha '"$ALPHA"
              '--lamda '"$LAMDA")
        echo 'python '"$SCRIPT"' '"${ARGS[@]}" >> tasks.sh
    done
done

# split tasks into files
perl -MList::Util=shuffle -e 'print shuffle(<STDIN>);' < tasks.sh > temp.sh
rm tasks.sh 2>/dev/null
split -l $TASKS_PER_FILE -a 2 temp.sh
rm temp.sh
AL=({a..z})
PREFIX='tasks_'
for i in `seq 0 25`; do
    for j in `seq 0 25`; do
        ID=$((i * 26 + j))
        ID=${ID##+(0)}
        mv 'x'"${AL[i]}${AL[j]}" "$PREFIX""$ID"'.sh' 2>/dev/null
        chmod +x "$PREFIX""$ID"'.sh' 2>/dev/null
    done
done