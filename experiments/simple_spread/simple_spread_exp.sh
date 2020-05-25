#!/usr/bin/bash

parallel -j 5 "python ../../main.py --scenario simple_spread --discrete --good-policy ddpg --save-dir ddpg-run-{}" ::: {1..10}

parallel -j 5 "python ../../main.py --scenario simple_spread --discrete --good-policy maddpg --save-dir maddpg-run-{}" ::: {1..10}

