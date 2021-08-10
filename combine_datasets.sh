#!/bin/bash

cd ./CUB_200_2011/CUB_200_2011/CUB_200_2011/images
#cd images
pwd
echo $$

# Rename CUB Image dataset to match the format of the 275 bird species dataset, keep underscores for splitting replace("_"," ")
python3 -c 'import os, shutil;list(map(lambda x: shutil.move(x, x[4:].upper()),os.listdir(".")))'

mkdir ../test ../train ../valid

valid_train_size=5

# Sends 5 random images to valid and test, then moves remaining images to train
for d in */; do
    images=($d/*.jpg)
    mkdir ../test/$d ../valid/$d
    i=1
    while [ ${i} -le ${valid_train_size} ]
    do
	mv ${images[RANDOM % ${#images[@]}]} ../test/$d
	images=($d/*.jpg)
	((i++))
    done
    i=1
    while [ ${i} -le ${valid_train_size} ]
    do
	mv ${images[RANDOM % ${#images[@]}]} ../valid/$d
	images=($d/*.jpg)
	((i++))
    done
    mv $d ../train/
done
echo "Train-valid-test split completed!"

cd ../train
python3 -c 'import os, shutil;list(map(lambda x: shutil.move(x, x.replace("_"," ")),os.listdir(".")))'
cd ../test
python3 -c 'import os, shutil;list(map(lambda x: shutil.move(x, x.replace("_"," ")),os.listdir(".")))'
cd ../valid
python3 -c 'import os, shutil;list(map(lambda x: shutil.move(x, x.replace("_"," ")),os.listdir(".")))'
cd ..

echo "Succesfully formatted CUB dataset to match 275 bird species dataset"

# Combine datasets
mv -n train/* ../../../275_Bird_Species/birds/train
mv -n test/* ../../../275_Bird_Species/birds/test
mv -n valid/* ../../../275_Bird_Species/birds/valid

echo "Datasets combined."
