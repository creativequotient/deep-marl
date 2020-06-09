#!/bin/bash

# Hyperparameters

runs=$(seq 0 10)
perturbation_factors=(0.0 0.1 0.01 0.001)

echo ${runs[@]}
echo ${perturbation_factors[@]}

# Run experiments

parallel -j 5 "python ../main.py --scenario simple_adversary --discrete --good-policy m3ddpg --adv-policy {1} --save-dir simple_adversary/{1}-vs-m3ddpg-run-{3}_perturbation-{2} --perturbation {2}" ::: ddpg m3ddpg ::: ${perturbation_factors[@]} ::: ${runs[@]}

parallel -j 5 "python ../main.py --scenario simple_push --discrete --good-policy m3ddpg --adv-policy {1} --save-dir simple_push/m3ddpg-vs-{1}-run-{3}_perturbation-{2} --perturbation {2}" ::: ddpg m3ddpg ::: ${perturbation_factors[@]} ::: ${runs[@]}

parallel -j 5 "python ../main.py --scenario simple_push --discrete --good-policy {1} --adv-policy m3ddpg --save-dir simple_push/{1}-vs-m3ddpg-run-{3}_perturbation-{2} --perturbation {2}" ::: ddpg m3ddpg ::: ${perturbation_factors[@]} ::: ${runs[@]}

parallel -j 5 "python ../main.py --scenario simple_speaker_listener --discrete --good-policy m3ddpg --save-dir simple_speaker_listener/m3ddpg-run-{2}_perturbation-{1} --perturbation {1}" ::: ${perturbation_factors[@]} ::: ${runs[@]}

parallel -j 5 "python ../main.py --scenario simple_spread --discrete --good-policy m3ddpg --save-dir simple_spread/m3ddpg-run-{2}_perturbation-{1} --perturbation {1}" ::: ${perturbation_factors[@]} ::: ${runs[@]}

parallel -j 5 "python ../main.py --scenario simple_spread --discrete --good-policy m3ddpg --save-dir simple_spread_128_units/m3ddpg-run-{2}_perturbation-{1} --perturbation {1} --num-units 128" ::: ${perturbation_factors[@]} ::: ${runs[@]}

parallel -j 5 "python ../main.py --scenario simple_tag --discrete --good-policy {1} --adv-policy m3ddpg --save-dir simple_tag/{1}-vs-m3ddpg-run-{3}_perturbation-{2} --perturbation {2}" ::: ddpg m3ddpg ::: ${perturbation_factors[@]} ::: ${runs[@]}

parallel -j 5 "python ../main.py --scenario simple_tag --discrete --good-policy {1} --adv-policy m3ddpg --save-dir simple_tag_128_units/{1}-vs-m3ddpg-run-{3}_perturbation-{2} --perturbation {2} --num-units 128" ::: ddpg m3ddpg ::: ${perturbation_factors[@]} ::: ${runs[@]}

# Run analysis

# parallel -j 5 "python ../evaluate.py --load-dir simple_tag/{1}-vs-{2}-run-{3} --eval-episodes 1000" ::: ddpg maddpg ::: ddpg maddpg ::: {1..10}

# parallel -j 5 "python ../evaluate.py --load-dir simple_tag_128_units/{1}-vs-{2}-run-{3} --eval-episodes 1000" ::: ddpg maddpg ::: ddpg maddpg ::: {1..10}

# parallel -j 5 "python ../evaluate.py --load-dir simple_push/{1}-vs-{2}-run-{3} --eval-episodes 1000" ::: ddpg maddpg ::: ddpg maddpg ::: {1..10}

# parallel -j 5 "python ../evaluate.py --load-dir simple_spread/{1}-run-{2} --eval-episodes 1000" ::: ddpg maddpg ::: {1..10}

# parallel -j 5 "python ../evaluate.py --load-dir simple_spread_128_units/{1}-run-{2} --eval-episodes 1000" ::: ddpg maddpg ::: {1..10}

# parallel -j 5 "python ../evaluate.py --load-dir simple_speaker_listener/{1}-run-{2} --eval-episodes 1000" ::: ddpg maddpg ::: {1..10}

# parallel -j 5 "python ../evaluate.py --load-dir simple_adversary/{1}-vs-{2}-run-{3} --eval-episodes 1000" ::: ddpg maddpg ::: ddpg maddpg ::: {1..10}
