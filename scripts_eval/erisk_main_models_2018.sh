#!/bin/bash

echo "eRisk 2018 collection"

#================== All features ##########
echo "Aggregating the results"
#natural/detect
#random_forest
python2 app/evaluation/aggregate_results.py -path /projets/sig/mullah/nlp/depression/output/2018/bow/test/All/random_forest/natural/detect -wsource resources/eval/2018/writings-per-subject-all-test.txt
#log_reg
#python2 app/evaluation/aggregate_results.py -path /projets/sig/mullah/nlp/depression/output/2018/bow/test/All/log_reg/natural/detect -wsource resources/eval/2018/writings-per-subject-all-test.txt
#sgd_log
python2 app/evaluation/aggregate_results.py -path /projets/sig/mullah/nlp/depression/output/2018/bow/test/All/sgd_log/natural/detect -wsource resources/eval/2018/writings-per-subject-all-test.txt
#svc_linear
python2 app/evaluation/aggregate_results.py -path /projets/sig/mullah/nlp/depression/output/2018/bow/test/All/svc_linear/natural/detect -wsource resources/eval/2018/writings-per-subject-all-test.txt
#svm_rbf
python2 app/evaluation/aggregate_results.py -path /projets/sig/mullah/nlp/depression/output/2018/bow/test/All/svc_rbf/natural/detect -wsource resources/eval/2018/writings-per-subject-all-test.txt
#knn3
python2 app/evaluation/aggregate_results.py -path /projets/sig/mullah/nlp/depression/output/2018/bow/test/All/knn3/natural/detect -wsource resources/eval/2018/writings-per-subject-all-test.txt
#nn
python2 app/evaluation/aggregate_results.py -path /projets/sig/mullah/nlp/depression/output/2018/bow/test/All/nn_lbfgs/natural/detect -wsource resources/eval/2018/writings-per-subject-all-test.txt
echo "Done."

echo "Estimating eRisk at o 5"
#natural/detect
echo "random forest"
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/bow/test/All/random_forest/natural/detect/test_eRisk_depression_global.txt -o 5

#echo "log_reg"
#python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/bow/test/All/log_reg/natural/detect/test_eRisk_depression_global.txt -o 5

echo "sgd_log"
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/bow/test/All/sgd_log/natural/detect/test_eRisk_depression_global.txt -o 5

echo "svc_linear"
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/bow/test/All/svc_linear/natural/detect/test_eRisk_depression_global.txt -o 5

echo "svm_rbf"
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/bow/test/All/svc_rbf/natural/detect/test_eRisk_depression_global.txt -o 5

echo "knn3"
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/bow/test/All/knn3/natural/detect/test_eRisk_depression_global.txt -o 5

echo "nn"
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/bow/test/All/nn_lbfgs/natural/detect/test_eRisk_depression_global.txt -o 5
echo "Done."

echo "Estimating eRisk at o 50"
#natural/detect
echo "random forest"
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/bow/test/All/random_forest/natural/detect/test_eRisk_depression_global.txt -o 50

#echo "log_reg"
#python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/bow/test/All/log_reg/natural/detect/test_eRisk_depression_global.txt -o 50

echo "sgd_log"
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/bow/test/All/sgd_log/natural/detect/test_eRisk_depression_global.txt -o 50

echo "svc_linear"
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/bow/test/All/svc_linear/natural/detect/test_eRisk_depression_global.txt -o 50

echo "svm_rbf"
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/bow/test/All/svc_rbf/natural/detect/test_eRisk_depression_global.txt -o 50

echo "knn3"
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/bow/test/All/knn3/natural/detect/test_eRisk_depression_global.txt -o 50

echo "nn"
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/bow/test/All/nn_lbfgs/natural/detect/test_eRisk_depression_global.txt -o 50
echo "Done."

