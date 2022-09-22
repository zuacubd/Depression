#!/bin/bash

#mean max
bash bert_evaluator.sh detect 2017 balanced mean max 0
bash bert_evaluator.sh detect 2017 balanced mean max 1
bash bert_evaluator.sh detect 2018 balanced mean max 0
bash bert_evaluator.sh detect 2018 balanced mean max 1

#mean mean
bash bert_evaluator.sh detect 2017 balanced mean mean 0
bash bert_evaluator.sh detect 2017 balanced mean mean 1
bash bert_evaluator.sh detect 2018 balanced mean mean 0
bash bert_evaluator.sh detect 2018 balanced mean mean 1

