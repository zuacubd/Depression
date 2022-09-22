#!/bin/bash

echo "eRisk 2017 collection"
#doc2vec features
#================== All features ##########
echo "Aggregating the results"
#natural/cumulative
#bert
python2 app/evaluation/aggregate_results.py -path  /projets/sig/mullah/nlp/depression/output/2017/bert/test/base_uncased/finetuned/natural/cumulative/detect -wsource resources/eval/2017/writings-per-subject-all-test.txt
echo "Done."

echo "Estimating eRisk at o 5"
#natural/cumulative
echo "bert"
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2017/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2017/bert/test/base_uncased/finetuned/natural/cumulative/detect/test_eRisk_depression_global.txt -o 5
echo "Done."


