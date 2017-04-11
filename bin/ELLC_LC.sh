#!/bin/bash
/Applications/MATLAB_R2015b.app/bin/matlab -nosplash -nodisplay -r "run ../matlab_scripts/makefirstsamplefile.m;quit;" 
ELLC LC ../config.txt
count=1
while [ 1 ]
do
	echo “in while loop”
	echo $count
	if [ $count -eq 1 ]
	then
		echo “Running small rotavg for first time”
		/Applications/MATLAB_R2015b.app/bin/matlab -nosplash -nodisplay -r " run ../matlab_scripts/small_batch_rotavg_bootstrap.m; quit"
	else
		echo “Running small avg”
		/Applications/MATLAB_R2015b.app/bin/matlab -nosplash -nodisplay -r " run ../matlab_scripts/small_batch_rotavg.m; quit"
	fi
	ELLC LC ../config.txt
	((count++))
done
