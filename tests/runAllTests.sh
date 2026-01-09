scripts=( "00_probe.py" "01_potentials.py" "02_propagate_otf=False.py" "02_propagate_otf=True.py" "03_manyprobes.py" "04_haadf.py" "05_tacaw.py" "05_tacaw_chunkFFT.py" "05_tacaw_cropped.py" "06_loaders.py" "07_defocus.py" "08_LACBED_iterative.py" "08_LACBED_onthefly.py" "10_midgley.py" "11_SED.py" "12_aberrations.py" )

echo $(date) > runAllTests-$1.log

for s in ${scripts[@]}
do
	echo python3 $s >> runAllTests-$1.log
	python3 $s >> runAllTests-$1.log 2>> runAllTests-$1.log
        #rm -rf psi_data/*
done
