scripts=( "00_probe.py"
	"01_potentials.py"
	"02_propagate_otf=False.py"
	"02_propagate_otf=True.py"
	"03_manyprobes.py"
	"04_haadf.py"
	"05_tacaw.py"
	"05_tacaw_chunkFFT.py"
	"05_tacaw_cropped.py"
	"06_loaders.py"
	"07_defocus.py"
	"08_LACBED_iterative.py"
	"08_LACBED_onthefly.py"
	"10_midgley.py"
	"11_SED.py"
	"12_aberrations.py"
	"04_haadf_otf.py"
	"05_tacaw_otf.py" )

echo $(date) > runAllTests-$1.log

mkdir ../$1

for s in ${scripts[@]}
do
	cp $s ../$1/
	cd ../$1
	ln -s ../tests/inputs inputs
	ln -s ../tests/outputs outputs
	echo python3 $s >> ../tests/runAllTests-$1.log
	python3 $s >> ../tests/runAllTests-$1.log 2>> ../tests/runAllTests-$1.log
        rm -rf psi_data/*
	rm $s
	cd ../tests
done

rmdir ../$1
