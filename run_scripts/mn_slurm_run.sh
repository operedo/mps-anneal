#!/bin/bash
# @ job_name         = mps-anneal
# @ initialdir       = .
# @ output           = exe_mps-anneal
# @ error            = err_mps-anneal
# @ total_tasks      = 3
# @ wall_clock_limit = 23:59:59 

set -- $(date "+%Y %m %d %H %M %S %Z")
year=$1
month=$2
day=$3
hour=$4
min=$5
sec=$6
tz=$7

timestamp="${month}${day}${hour}${min}"

home=/home/bsc21/bsc21021

tag="canal_sin"
#tag="ellipsim0_sin"
#tag="ellipsim45_sin"

NSLOTS=3
nx=100
ny=100
nz=1
tnx=5
tny=5
tnz=1
scenario="AA"
hashsize=400000
t0=5000
#t0=7500
lambda=0.1
#lambda=0.15
npert=$[10000 * $nx * $ny *$nz]
ntemp=10
maxatt=5
irepo=$[1*$nx*$ny*$nz]
dataTI="${home}/mps-anneal/resources/test${nx}x${ny}x${nz}.dat"
#dataTI="${home}/mps-anneal/resources/ellipsim${nx}x${ny}x${nz}_15_5_0_0.dat"
#dataTI="${home}/mps-anneal/resources/ellipsim${nx}x${ny}x${nz}_15_5_45_0.dat"
dataRE="${home}/mps-anneal/resources/randomimage${nx}x${ny}x${nz}.dat"
logTI="${home}/mps-anneal/resources/logTI_${nx}x${ny}x${nz}_${tnx}x${tny}x${tnz}_${NSLOTS}_${scenario}_${timestamp}_${t0}_${tag}.txt"
logRE="${home}/mps-anneal/resources/logRE_${nx}x${ny}x${nz}_${tnx}x${tny}x${tnz}_${NSLOTS}_${scenario}_${timestamp}_${t0}_${tag}.txt"
topo="${home}/mps-anneal/resources/topology${NSLOTS}.dat"
out="${home}/mps-anneal/resources/${tnx}x${tny}x${tnz}_${NSLOTS}_${scenario}_${timestamp}_${t0}_${tag}.dat"
echo "$out"
sched="${home}/mps-anneal/resources/schedule${nx}x${ny}x${nz}_${tnx}x${tny}x${tnz}_${NSLOTS}_${scenario}_${timestamp}_${t0}_${tag}.dat"
rand="${home}/mps-anneal/resources/randomfile${nx}x${ny}x${nz}_${scenario}.dat"
rand2="${home}/mps-anneal/resources/randomfile${nx}x${ny}x${nz}_${scenario}_2.dat"

advance="salida${tnx}x${tny}x${tnz}_${NSLOTS}_${scenario}_${timestamp}_${t0}_${tag}.dat"
echo "$advance"

echo "srun ./mps-anneal_sin $dataTI $dataRE $logTI $logRE $topo $out $sched $rand $rand2 $nx $ny $nz $tnx $tny $tnz $hashsize $t0 $lambda $npert $ntemp $maxatt $irepo > outputs/$advance"

srun ./mps-anneal_sin $dataTI $dataRE $logTI $logRE $topo $out $sched $rand $rand2 $nx $ny $nz $tnx $tny $tnz $hashsize $t0 $lambda $npert $ntemp $maxatt $irepo >> outputs/$advance

#echo $HOSTNAME >> outputs/$advance

exit 0
