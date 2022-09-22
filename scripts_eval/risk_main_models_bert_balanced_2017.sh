#!/bin/bash

echo "eRisk 2017 collection"
#doc2vec features
#================== All features ##########
echo "Aggregating the results"
#balanced
#bert
python2 app/evaluation/aggregate_results.py -path  /projets/sig/mullah/nlp/depression/output/2017/bert/test/base_uncased/finetuned/balanced/detect -wsource resources/eval/2017/writings-per-subject-all-test.txt
echo "Done."

echo "Estimating eRisk at o 5"
#balanced
echo "bert"
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2017/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2017/bert/test/base_uncased/finetuned/balanced/detect/test_eRisk_depression_global.txt -o 5
echo "Done."


