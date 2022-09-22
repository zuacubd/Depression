#!/bin/bash

#mean-max
bash bert_evaluator.sh detect 2017 natural mean max 0
bash bert_evaluator.sh detect 2017 natural mean max 1
bash bert_evaluator.sh detect 2018 natural mean max 0
bash bert_evaluator.sh detect 2018 natural mean max 1

#mean-mean
bash bert_evaluator.sh detect 2017 natural mean mean 0
bash bert_evaluator.sh detect 2017 natural mean mean 1
bash bert_evaluator.sh detect 2018 natural mean mean 0
bash bert_evaluator.sh detect 2018 natural mean mean 1

