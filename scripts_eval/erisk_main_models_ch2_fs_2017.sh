#!/bin/bash

echo "eRisk 2017 collection"

#ch2_fs features
#================== All features ##########
echo "Aggregating the results"
#natural/early_detect
#random_forest
python2 app/evaluation/aggregate_results.py -path /projets/sig/mullah/nlp/depression/output/2017/ch2_fs/test/All/random_forest/natural/early_detect -wsource resources/eval/2017/writings-per-subject-all-test.txt
#log_reg
#python2 app/evaluation/aggregate_results.py -path /projets/sig/mullah/nlp/depression/output/2017/ch2_fs/test/All/log_reg/natural/early_detect -wsource resources/eval/2017/writings-per-subject-all-test.txt
#sgd_log
python2 app/evaluation/aggregate_results.py -path /projets/sig/mullah/nlp/depression/output/2017/ch2_fs/test/All/sgd_log/natural/early_detect -wsource resources/eval/2017/writings-per-subject-all-test.txt
#svc_linear
python2 app/evaluation/aggregate_results.py -path /projets/sig/mullah/nlp/depression/output/2017/ch2_fs/test/All/svc_linear/natural/early_detect -wsource resources/eval/2017/writings-per-subject-all-test.txt
#svm_rbf
python2 app/evaluation/aggregate_results.py -path /projets/sig/mullah/nlp/depression/output/2017/ch2_fs/test/All/svc_rbf/natural/early_detect -wsource resources/eval/2017/writings-per-subject-all-test.txt
#knn3
python2 app/evaluation/aggregate_results.py -path /projets/sig/mullah/nlp/depression/output/2017/ch2_fs/test/All/knn3/natural/early_detect -wsource resources/eval/2017/writings-per-subject-all-test.txt
#nn
python2 app/evaluation/aggregate_results.py -path /projets/sig/mullah/nlp/depression/output/2017/ch2_fs/test/All/nn_lbfgs/natural/early_detect -wsource resources/eval/2017/writings-per-subject-all-test.txt
echo "Done."

echo "Estimating eRisk at o 5"
#natural/early_detect
echo "random forest"
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2017/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2017/ch2_fs/test/All/random_forest/natural/early_detect/test_eRisk_depression_global.txt -o 5

#echo "log_reg"
#python2 app/evaluation/erisk_eval.py -gpath resources/eval/2017/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2017/ch2_fs/test/All/log_reg/natural/early_detect/test_eRisk_depression_global.txt -o 5

echo "sgd_log"
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2017/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2017/ch2_fs/test/All/sgd_log/natural/early_detect/test_eRisk_depression_global.txt -o 5

echo "svc_linear"
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2017/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2017/ch2_fs/test/All/svc_linear/natural/early_detect/test_eRisk_depression_global.txt -o 5

echo "svm_rbf"
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2017/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2017/ch2_fs/test/All/svc_rbf/natural/early_detect/test_eRisk_depression_global.txt -o 5

echo "knn3"
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2017/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2017/ch2_fs/test/All/knn3/natural/early_detect/test_eRisk_depression_global.txt -o 5

echo "nn"
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2017/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2017/ch2_fs/test/All/nn_lbfgs/natural/early_detect/test_eRisk_depression_global.txt -o 5
echo "Done."

echo "Estimating eRisk at o 50"
#natural/early_detect
echo "random forest"
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2017/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2017/ch2_fs/test/All/random_forest/natural/early_detect/test_eRisk_depression_global.txt -o 50

#echo "log_reg"
#python2 app/evaluation/erisk_eval.py -gpath resources/eval/2017/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2017/ch2_fs/test/All/log_reg/natural/early_detect/test_eRisk_depression_global.txt -o 50

echo "sgd_log"
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2017/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2017/ch2_fs/test/All/sgd_log/natural/early_detect/test_eRisk_depression_global.txt -o 50

echo "svc_linear"
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2017/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2017/ch2_fs/test/All/svc_linear/natural/early_detect/test_eRisk_depression_global.txt -o 50

echo "svm_rbf"
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2017/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2017/ch2_fs/test/All/svc_rbf/natural/early_detect/test_eRisk_depression_global.txt -o 50

echo "knn3"
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2017/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2017/ch2_fs/test/All/knn3/natural/early_detect/test_eRisk_depression_global.txt -o 50

echo "nn"
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2017/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2017/ch2_fs/test/All/nn_lbfgs/natural/early_detect/test_eRisk_depression_global.txt -o 50
echo "Done."

