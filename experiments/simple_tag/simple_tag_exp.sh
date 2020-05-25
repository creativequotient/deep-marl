#!/usr/bin/bash

parallel -j 5 "python ../../main.py --scenario simple_tag --discrete --num-adversaries 3 --good-policy ddpg --adv-policy ddpg --save-dir ddpg-vs-ddpg-run-{}" ::: {1..10}

parallel -j 5 "python ../../main.py --scenario simple_tag --discrete --num-adversaries 3 --good-policy ddpg --adv-policy maddpg --save-dir ddpg-vs-maddpg-run-{}" ::: {1..10}

parallel -j 5 "python ../../main.py --scenario simple_tag --discrete --num-adversaries 3 --good-policy maddpg --adv-policy ddpg --save-dir maddpg-vs-ddpg-run-{}" ::: {1..10}

parallel -j 5 "python ../../main.py --scenario simple_tag --discrete --num-adversaries 3 --good-policy maddpg --adv-policy maddpg --save-dir maddpg-vs-maddpg-run-{}" ::: {1..10}

