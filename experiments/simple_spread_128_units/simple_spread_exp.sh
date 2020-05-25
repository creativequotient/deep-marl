#!/usr/bin/bash

parallel -j 5 "python ../../main.py --scenario simple_spread --num-units 128 --discrete --good-policy ddpg --save-dir ddpg-run-{}" ::: {1..10}

parallel -j 5 "python ../../main.py --scenario simple_spread --num-units 128 --discrete --good-policy maddpg --save-dir maddpg-run-{}" ::: {1..10}

