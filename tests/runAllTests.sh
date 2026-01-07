scripts=(  "06_loaders.py" "07_defocus.py" "08_LACBED_iterative.py" "08_LACBED_onthefly.py" "10_midgley.py" "11_SED.py" "12_aberrations.py" )

echo $(date) > runAllTests.log

for s in ${scripts[@]}
do
	for version in 3 3.13 3.9
	do
		echo python$version $s >> runAllTests.log
		python$version $s >> runAllTests.log 2>> runAllTests.log
        rm -rf psi_data/*
	done
done
