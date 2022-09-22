#!/bin/bash

echo "eRisk 2018 collection"
#doc2vec features
#================== All features ##########
echo "Aggregating the results"
#natural/mean/mean
#bert
python2 app/evaluation/aggregate_results.py -path  /projets/sig/mullah/nlp/depression/output/2018/bert/test/base_uncased/finetuned/natural/mean/mean/early_detect -wsource resources/eval/2018/writings-per-subject-all-test.txt
echo "Done."

echo "Estimating eRisk at o 5"
#natural/mean/mean
echo "bert"
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/bert/test/base_uncased/finetuned/natural/mean/mean/early_detect/test_eRisk_depression_global.txt -o 5
echo "Done."

echo "Estimating eRisk at o 50"
#natural/mean/mean
echo "bert"
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/bert/test/base_uncased/finetuned/natural/mean/mean/early_detect/test_eRisk_depression_global.txt -o 50
echo "Done."


