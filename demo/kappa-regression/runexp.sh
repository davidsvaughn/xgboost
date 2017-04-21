#!/bin/bash

# if [ -f YearPredictionMSD.txt ]
# then
#     echo "use existing data to run experiment"
# else
#     echo "getting data from uci, make sure you are connected to internet"
#     wget https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip
#     unzip YearPredictionMSD.txt.zip
# fi
# map feature using indicator encoding, also produce featmap.txt

#========================================================================
if [ $1 -eq 1 ]
then
	echo "massage data..."
	python csv2libsvm.py train.csv train.libsvm -1 1
	python csv2libsvm.py test.csv test.libsvm -2 1
fi


#========================================================================
echo "shuffle data..."
randseed()
{
  seed="$1"
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
    </dev/zero 2>/dev/null
}
SEED=$RANDOM
#SEED=24055
#echo "SEED==$SEED"
shuf train.libsvm --random-source=<(randseed $SEED) | sponge train.libsvm
## test -->
#shuf train.libsvm --random-source=<(randseed $SEED) | head -1 > tmp.libsvm


#========================================================================
echo "split data..."
head -50000 train.libsvm > trn.libsvm
tail -n +50001 train.libsvm > val.libsvm


#========================================================================
echo "train model..."
../../xgboost train.conf

echo "generate test predictions..."
../../xgboost test.conf

#========================================================================
# cleanup...
rm model.bin
rm *.buffer