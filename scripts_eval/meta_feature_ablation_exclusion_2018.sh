#!/bin/bash

#natural
#2018
#aggregation
#random forest
echo "Aggregation"
python2 app/evaluation/aggregate_results.py -path /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_meta1_exclusion/random_forest/natural -wsource resources/eval/2018/writings-per-subject-all-test.txt
python2 app/evaluation/aggregate_results.py -path /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_meta2_exclusion/random_forest/natural -wsource resources/eval/2018/writings-per-subject-all-test.txt
python2 app/evaluation/aggregate_results.py -path /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_meta3_exclusion/random_forest/natural -wsource resources/eval/2018/writings-per-subject-all-test.txt
python2 app/evaluation/aggregate_results.py -path /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_meta4_exclusion/random_forest/natural -wsource resources/eval/2018/writings-per-subject-all-test.txt
python2 app/evaluation/aggregate_results.py -path /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_meta5_exclusion/random_forest/natural -wsource resources/eval/2018/writings-per-subject-all-test.txt
python2 app/evaluation/aggregate_results.py -path /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_meta6_exclusion/random_forest/natural -wsource resources/eval/2018/writings-per-subject-all-test.txt
echo "Done"

echo "Estimating erisk 5"
echo "Meta1: "
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_meta1_exclusion/random_forest/natural/test_eRisk_depression_global.txt -o 5

echo "Meta2: "
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_meta2_exclusion/random_forest/natural/test_eRisk_depression_global.txt -o 5

echo "Meta3: "
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_meta3_exclusion/random_forest/natural/test_eRisk_depression_global.txt -o 5

echo "Meta4: "
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_meta4_exclusion/random_forest/natural/test_eRisk_depression_global.txt -o 5

echo "Meta5: "
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_meta5_exclusion/random_forest/natural/test_eRisk_depression_global.txt -o 5

echo "Meta6: "
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_meta6_exclusion/random_forest/natural/test_eRisk_depression_global.txt -o 5


echo "Estimating erisk 50"

echo "Meta1: "
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_meta1_exclusion/random_forest/natural/test_eRisk_depression_global.txt -o 50

echo "Meta2: "
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_meta2_exclusion/random_forest/natural/test_eRisk_depression_global.txt -o 50

echo "Meta3: "
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_meta3_exclusion/random_forest/natural/test_eRisk_depression_global.txt -o 50

echo "Meta4: "
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_meta4_exclusion/random_forest/natural/test_eRisk_depression_global.txt -o 50

echo "Meta5: "
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_meta5_exclusion/random_forest/natural/test_eRisk_depression_global.txt -o 50

echo "Meta6: "
python2 app/evaluation/erisk_eval.py -gpath resources/eval/2018/risk-golden-truth-test.txt -ppath /projets/sig/mullah/nlp/depression/output/2018/bow/test/All_meta6_exclusion/random_forest/natural/test_eRisk_depression_global.txt -o 50

echo "Done."
