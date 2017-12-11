#!/bin/sh
echo "Generating Event Features..."
python feature_vector_generator.py
echo "Classifying Events..."
python decisionTree.py

for k in 1 2 3 4
do
	cp "$(dirname $PWD)/data/out/rf_predict_class_$k.vectors" "$(dirname $PWD)/data/out/predict_class_$k.vectors"
done

echo "Clustering events..."
python Cos_DBScan.py
echo "Ranking clusters..."
python clusterRanking.py
