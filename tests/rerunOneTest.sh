s="04_haadf_otf.py"
s="21_memorytests.py"

echo $(date) >> runAllTests-$1.log

mkdir ../$1

cp $s ../$1/
cd ../$1
ln -s ../tests/inputs inputs
ln -s ../tests/outputs outputs
echo python3 $s >> ../tests/runAllTests-$1.log
python3 $s >> ../tests/runAllTests-$1.log 2>> ../tests/runAllTests-$1.log
rm -rf psi_data/*
rm $s
cd ../tests

rmdir ../$1
