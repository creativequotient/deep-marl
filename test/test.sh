#!/bin/bash

parallel -j 4 "../main.py --scenario simple_speaker_listener --good-policy m3ddpg --save-dir ./simple_speaker_listener/m3ddpg_run_{2} --discrete --perturbation {1}" ::: 0.0 0.1 0.01 0.001 ::: {1..8}

parallel -j 4 "../main.py --scenario simple_spread --good-policy m3ddpg --save-dir ./simple_spread_{3}_units/m3ddpg_run_{2} --discrete --perturbation {1} --num-units {3}" ::: 0.0 0.1 0.01 0.001 ::: {1..8} ::: 64 128

parallel -j 4 "python ../evaluate.py --load-dir ./simple_speaker_listener/m3ddpg-run-{1} --eval-episodes 1000" ::: {1..8}

parallel -j 4 "python ../evaluate.py --load-dir ./simple_spread_{2}_units/m3ddpg_run_{1} --eval-episodes 1000" ::: {1..8} ::: 64 128