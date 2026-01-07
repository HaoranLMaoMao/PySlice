scripts=( "00_probe.py" "01_potentials.py" "02_propagate_otf=False.py" "02_propagate_otf=True.py" "03_manyprobes.py" "04_haadf.py" "05_tacaw.py" "05_tacaw_chunkFFT.py" "05_tacaw_cropped.py" "06_loaders.py" "07_defocus.py" "08_LACBED_iterative.py" "08_LACBED_onthefly.py" "10_midgley.py" "11_SED.py" "12_aberrations.py" )

echo $(date) > runAllTests.log

mv /media/qwe/Alexandria/Software/Python3.12/site-packages/torch_bak /media/qwe/Alexandria/Software/Python3.12/site-packages/torch

for s in ${scripts[@]}
do
	echo python3 $s >> runAllTests.log
	python3 $s >> runAllTests.log 2>> runAllTests.log
        rm -rf psi_data/*
done

mv /media/qwe/Alexandria/Software/Python3.12/site-packages/torch /media/qwe/Alexandria/Software/Python3.12/site-packages/torch_bak

for s in ${scripts[@]}
do
	echo python3 $s >> runAllTests.log
	python3 $s >> runAllTests.log 2>> runAllTests.log
        rm -rf psi_data/*
done

mv /media/qwe/Alexandria/Software/Python3.12/site-packages/torch_bak /media/qwe/Alexandria/Software/Python3.12/site-packages/torch
