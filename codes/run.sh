#!/bin/sh
echo "PYTHON VERSION 2.7"
pyver=`python --version 2>&1 | grep '2.7'`
if [ "$pyver" == "" ];then
	echo "CURRENT PYTHON VERSION IS NOT 2.7, ABORTING"
	exit
fi

echo ""
echo "Checking Required Packages..."
echo ""
echo "Checking sklearn..."
check=`python -m pip show sklearn`
if [ "$check" == "" ];then
	echo "sklearn is not installed, installing..."
	python -m pip install sklearn --user
else
	echo "sklearn has been installed"
fi
echo ""

echo "Checking nltk..."
check=`python -m pip show nltk`
if [ "$check" == "" ];then
	echo "nltk is not installed, installing..."
	python -m pip install nltk --user
	python -m nltk.downloader all
else
	echo "nltk has been installed"
fi
echo ""

echo "Checking numpy..."
check=`python -m pip show numpy`
if [ "$check" == "" ];then
	echo "numpy is not installed, installing..."
	python -m pip install numpy --user
else
	echo "numpy has been installed"
fi
echo ""

echo "Checking matplotlib..."
check=`python -m pip show matplotlib`
if [ "$check" == "" ];then
	echo "matplotlib is not installed, installing..."
	python -m pip install matplotlib --user
else
	echo "matplotlib has been installed"
fi
echo ""

echo "Checking tweepy..."
check=`python -m pip show tweepy`
if [ "$check" == "" ];then
	echo "tweepy is not installed, installing..."
	python -m pip install tweepy --user
else
	echo "tweepy has been installed"
fi
echo ""

echo "Checking pytorch..."
check=`python -m pip show pytorch`
if [ "$check" == "" ];then
	echo "pytorch is not installed, installing..."
	python -m pip install pytorch --user
else
	echo "pytorch has been installed"
fi
echo ""

echo "Generating Event Features..."
python feature_vector_generator.py
echo ""

echo "Classifying Events..."
python decisionTree.py
echo ""

for k in 1 2 3 4
do
	cp "$(dirname $PWD)/data/out/rf_predict_class_$k.vectors" "$(dirname $PWD)/data/out/predict_class_$k.vectors"
done

echo "Clustering events..."
python Cos_DBScan.py
echo ""

echo "Ranking clusters..."
python clusterRanking.py
echo ""

for k in 1 2 3 4
do
	rm "$(dirname $PWD)/data/out/predict_class_$k.vectors"
done