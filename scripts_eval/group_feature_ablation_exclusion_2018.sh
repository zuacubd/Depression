#!/bin/bash

#natural
#2018
#aggregation
#random forest
echo "Aggregation"
python2 app/evaluation/aggregate_results.py -path /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_g1_exclusion/random_forest/natural -wsource resources/eval/2018/writings-per-subject-all-test.txt
python2 app/evaluation/aggregate_results.py -path /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_g2_exclusion/random_forest/natural -wsource resources/eval/2018/writings-per-subject-all-test.txt
python2 app/evaluation/aggregate_results.py -path /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_g3_exclusion/random_forest/natural -wsource resources/eval/2018/writings-per-subject-all-test.txt
python2 app/evaluation/aggregate_results.py -path /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_g4_exclusion/random_forest/natural -wsource resources/eval/2018/writings-per-subject-all-test.txt
python2 app/evaluation/aggregate_results.py -path /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_g5_exclusion/random_forest/natural -wsource resources/eval/2018/writings-per-subject-all-test.txt
python2 app/evaluation/aggregate_results.py -path /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_g6_exclusion/random_forest/natural -wsource resources/eval/2018/writings-per-subject-all-test.txt
python2 app/evaluation/aggregate_results.py -path /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_g7_exclusion/random_forest/natural -wsource resources/eval/2018/writings-per-subject-all-test.txt
python2 app/evaluation/aggregate_results.py -path /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_g8_exclusion/random_forest/natural -wsource resources/eval/2018/writings-per-subject-all-test.txt
python2 app/evaluation/aggregate_results.py -path /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_g9_exclusion/random_forest/natural -wsource resources/eval/2018/writings-per-subject-all-test.txt
python2 app/evaluation/aggregate_results.py -path /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_g10_exclusion/random_forest/natural -wsource resources/eval/2018/writings-per-subject-all-test.txt
python2 app/evaluation/aggregate_results.py -path /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_g11_exclusion/random_forest/natural -wsource resources/eval/2018/writings-per-subject-all-test.txt
python2 app/evaluation/aggregate_results.py -path /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_g12_exclusion/random_forest/natural -wsource resources/eval/2018/writings-per-subject-all-test.txt
python2 app/evaluation/aggregate_results.py -path /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_g13_exclusion/random_forest/natural -wsource resources/eval/2018/writings-per-subject-all-test.txt
python2 app/evaluation/aggregate_results.py -path /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_g14_exclusion/random_forest/natural -wsource resources/eval/2018/writings-per-subject-all-test.txt
python2 app/evaluation/aggregate_results.py -path /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_g15_exclusion/random_forest/natural -wsource resources/eval/2018/writings-per-subject-all-test.txt
python2 app/evaluation/aggregate_results.py -path /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_g16_exclusion/random_forest/natural -wsource resources/eval/2018/writings-per-subject-all-test.txt
echo "Done"

echo "Estimating erisk 5"
echo "G1: "
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_g1_exclusion/random_forest/natural/test_eRisk_depression_global.txt -o 5

echo "G2: "
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_g2_exclusion/random_forest/natural/test_eRisk_depression_global.txt -o 5

echo "G3: "
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_g3_exclusion/random_forest/natural/test_eRisk_depression_global.txt -o 5

echo "G4: "
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_g4_exclusion/random_forest/natural/test_eRisk_depression_global.txt -o 5

echo "G5: "
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_g5_exclusion/random_forest/natural/test_eRisk_depression_global.txt -o 5

echo "G6: "
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_g6_exclusion/random_forest/natural/test_eRisk_depression_global.txt -o 5

echo "G7: "
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_g7_exclusion/random_forest/natural/test_eRisk_depression_global.txt -o 5

echo "G8: "
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_g8_exclusion/random_forest/natural/test_eRisk_depression_global.txt -o 5

echo "G9: "
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_g9_exclusion/random_forest/natural/test_eRisk_depression_global.txt -o 5

echo "G10: "
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_g10_exclusion/random_forest/natural/test_eRisk_depression_global.txt -o 5

echo "G11: "
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_g11_exclusion/random_forest/natural/test_eRisk_depression_global.txt -o 5

echo "G12: "
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_g12_exclusion/random_forest/natural/test_eRisk_depression_global.txt -o 5

echo "G13: "
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_g13_exclusion/random_forest/natural/test_eRisk_depression_global.txt -o 5

echo "G14: "
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_g14_exclusion/random_forest/natural/test_eRisk_depression_global.txt -o 5

echo "G15: "
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_g15_exclusion/random_forest/natural/test_eRisk_depression_global.txt -o 5

echo "G16: "
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_g16_exclusion/random_forest/natural/test_eRisk_depression_global.txt -o 5


echo "Estimating erisk 50"

echo "G1: "
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_g1_exclusion/random_forest/natural/test_eRisk_depression_global.txt -o 50

echo "G2: "
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_g2_exclusion/random_forest/natural/test_eRisk_depression_global.txt -o 50

echo "G3: "
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_g3_exclusion/random_forest/natural/test_eRisk_depression_global.txt -o 50

echo "G4: "
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_g4_exclusion/random_forest/natural/test_eRisk_depression_global.txt -o 50

echo "G5: "
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_g5_exclusion/random_forest/natural/test_eRisk_depression_global.txt -o 50

echo "G6: "
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_g6_exclusion/random_forest/natural/test_eRisk_depression_global.txt -o 50

echo "G7: "
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_g7_exclusion/random_forest/natural/test_eRisk_depression_global.txt -o 50

echo "G8: "
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_g8_exclusion/random_forest/natural/test_eRisk_depression_global.txt -o 50

echo "G9: "
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_g9_exclusion/random_forest/natural/test_eRisk_depression_global.txt -o 50

echo "G10: "
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_g10_exclusion/random_forest/natural/test_eRisk_depression_global.txt -o 50

echo "G11: "
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_g11_exclusion/random_forest/natural/test_eRisk_depression_global.txt -o 50

echo "G12: "
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_g12_exclusion/random_forest/natural/test_eRisk_depression_global.txt -o 50

echo "G13: "
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_g13_exclusion/random_forest/natural/test_eRisk_depression_global.txt -o 50

echo "G14: "
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_g14_exclusion/random_forest/natural/test_eRisk_depression_global.txt -o 50

echo "G15: "
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_g15_exclusion/random_forest/natural/test_eRisk_depression_global.txt -o 50

echo "G16: "
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_g16_exclusion/random_forest/natural/test_eRisk_depression_global.txt -o 50
echo "Done."
