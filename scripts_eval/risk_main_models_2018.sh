#!/bin/bash

echo "eRisk 2018 collection"

#================== All features ##########
echo "Aggregating the results"
#natural
#random_forest
python2 app/evaluation/aggregate_results.py -path /projets/sig/mullah/nlp/depression/output/2018/test/All/random_forest/natural -wsource resources/eval/2018/writings-per-subject-all-test.txt
#log_reg
python2 app/evaluation/aggregate_results.py -path /projets/sig/mullah/nlp/depression/output/2018/test/All/log_reg/natural -wsource resources/eval/2018/writings-per-subject-all-test.txt
#sgd_log
python2 app/evaluation/aggregate_results.py -path /projets/sig/mullah/nlp/depression/output/2018/test/All/sgd_log/natural -wsource resources/eval/2018/writings-per-subject-all-test.txt
#svc_linear
python2 app/evaluation/aggregate_results.py -path /projets/sig/mullah/nlp/depression/output/2018/test/All/svc_linear/natural -wsource resources/eval/2018/writings-per-subject-all-test.txt
#svm_rbf
python2 app/evaluation/aggregate_results.py -path /projets/sig/mullah/nlp/depression/output/2018/test/All/svc_rbf/natural -wsource resources/eval/2018/writings-per-subject-all-test.txt
#knn3
python2 app/evaluation/aggregate_results.py -path /projets/sig/mullah/nlp/depression/output/2018/test/All/knn3/natural -wsource resources/eval/2018/writings-per-subject-all-test.txt
#nn
python2 app/evaluation/aggregate_results.py -path /projets/sig/mullah/nlp/depression/output/2018/test/All/nn_lbfgs/natural -wsource resources/eval/2018/writings-per-subject-all-test.txt
echo "Done."

echo "Estimating eRisk at o 5"
#natural
echo "random forest"
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/test/All/random_forest/natural/test_eRisk_depression_global.txt -o 5

echo "log_reg"
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/test/All/log_reg/natural/test_eRisk_depression_global.txt -o 5

echo "sgd_log"
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/test/All/sgd_log/natural/test_eRisk_depression_global.txt -o 5

echo "svc_linear"
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/test/All/svc_linear/natural/test_eRisk_depression_global.txt -o 5

echo "svm_rbf"
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/test/All/svc_rbf/natural/test_eRisk_depression_global.txt -o 5

echo "knn3"
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/test/All/knn3/natural/test_eRisk_depression_global.txt -o 5

echo "nn"
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/test/All/nn_lbfgs/natural/test_eRisk_depression_global.txt -o 5
echo "Done."

echo "Estimating eRisk at o 50"
#natural
echo "random forest"
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/test/All/random_forest/natural/test_eRisk_depression_global.txt -o 50

echo "log_reg"
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/test/All/log_reg/natural/test_eRisk_depression_global.txt -o 50

echo "sgd_log"
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/test/All/sgd_log/natural/test_eRisk_depression_global.txt -o 50

echo "svc_linear"
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/test/All/svc_linear/natural/test_eRisk_depression_global.txt -o 50

echo "svm_rbf"
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/test/All/svc_rbf/natural/test_eRisk_depression_global.txt -o 50

echo "knn3"
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/test/All/knn3/natural/test_eRisk_depression_global.txt -o 50

echo "nn"
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/test/All/nn_lbfgs/natural/test_eRisk_depression_global.txt -o 50
echo "Done."

