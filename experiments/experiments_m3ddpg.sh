#!/bin/bash

# Hyperparameters

runs=$(seq 1 10)
perturbation_factors=(0.0 0.001 0.01 0.1)

echo ${runs[@]}
echo ${perturbation_factors[@]}

if [ $1 == '1' ]
then
  echo "Running set 1"

  parallel -j 5 "python ../main.py --scenario simple_adversary --num-adversaries 1 --discrete --good-policy m3ddpg --adv-policy {1} --save-dir simple_adversary/{1}-vs-m3ddpg-run-{3}_perturbation-{2} --perturbation {2}" ::: ddpg m3ddpg ::: ${perturbation_factors[@]} ::: ${runs[@]}

  parallel -j 5 "python ../main.py --scenario simple_adversary --num-adversaries 1 --discrete --good-policy {1} --adv-policy ddpg --save-dir simple_adversary/{1}-vs-m3ddpg-run-{3}_perturbation-{2} --perturbation {2}" ::: ddpg m3ddpg ::: ${perturbation_factors[@]} ::: ${runs[@]}

  parallel -j 5 "python ../main.py --scenario simple_push --num-adversaries 1 --discrete --good-policy m3ddpg --adv-policy {1} --save-dir simple_push/m3ddpg-vs-{1}-run-{3}_perturbation-{2} --perturbation {2}" ::: ddpg m3ddpg ::: ${perturbation_factors[@]} ::: ${runs[@]}

  parallel -j 5 "python ../main.py --scenario simple_push --num-adversaries 1 --discrete --good-policy {1} --adv-policy m3ddpg --save-dir simple_push/{1}-vs-m3ddpg-run-{3}_perturbation-{2} --perturbation {2}" ::: ddpg m3ddpg ::: ${perturbation_factors[@]} ::: ${runs[@]}

  parallel -j 5 "python ../main.py --scenario simple_speaker_listener --discrete --good-policy m3ddpg --save-dir simple_speaker_listener/m3ddpg-run-{2}_perturbation-{1} --perturbation {1}" ::: ${perturbation_factors[@]} ::: ${runs[@]}
elif [ $1 == '2' ]
then
  echo "Running set 2"

  parallel -j 5 "python ../main.py --scenario simple_spread --discrete --good-policy m3ddpg --save-dir simple_spread/m3ddpg-run-{2}_perturbation-{1} --perturbation {1}" ::: ${perturbation_factors[@]} ::: ${runs[@]}

  parallel -j 5 "python ../main.py --scenario simple_spread --discrete --good-policy m3ddpg --save-dir simple_spread_128_units/m3ddpg-run-{2}_perturbation-{1} --perturbation {1} --num-units 128" ::: ${perturbation_factors[@]} ::: ${runs[@]}

  parallel -j 5 "python ../main.py --scenario simple_tag --num-adversaries 3 --discrete --good-policy {1} --adv-policy m3ddpg --save-dir simple_tag/{1}-vs-m3ddpg-run-{3}_perturbation-{2} --perturbation {2}" ::: ddpg m3ddpg ::: ${perturbation_factors[@]} ::: ${runs[@]}

  parallel -j 5 "python ../main.py --scenario simple_tag --num-adversaries 3 --discrete --good-policy {1} --adv-policy m3ddpg --save-dir simple_tag_128_units/{1}-vs-m3ddpg-run-{3}_perturbation-{2} --perturbation {2} --num-units 128" ::: ddpg m3ddpg ::: ${perturbation_factors[@]} ::: ${runs[@]}
elif [ $1 == 'eval' ]
then
    echo "Evaluating..."

    parallel -j 5 "python ../evaluate.py --eval-episodes 1000 --load-dir simple_adversary/{1}-vs-m3ddpg-run-{3}_perturbation-{2}" ::: ddpg m3ddpg ::: ${perturbation_factors[@]} ::: ${runs[@]}

    parallel -j 5 "python ../evaluate.py --eval-episodes 1000 --load-dir simple_adversary/{1}-vs-ddpg-run-{3}_perturbation-{2}" ::: ddpg m3ddpg ::: ${perturbation_factors[@]} ::: ${runs[@]}

    parallel -j 5 "python ../evaluate.py --eval-episodes 1000 --load-dir simple_push/m3ddpg-vs-{1}-run-{3}_perturbation-{2}" ::: ddpg m3ddpg ::: ${perturbation_factors[@]} ::: ${runs[@]}

    parallel -j 5 "python ../evaluate.py --eval-episodes 1000 --load-dir simple_push/{1}-vs-m3ddpg-run-{3}_perturbation-{2}" ::: ddpg m3ddpg ::: ${perturbation_factors[@]} ::: ${runs[@]}

    parallel -j 5 "python ../evaluate.py --eval-episodes 1000 --load-dir simple_speaker_listener/m3ddpg-run-{2}_perturbation-{1}" ::: ${perturbation_factors[@]} ::: ${runs[@]}

    parallel -j 5 "python ../evaluate.py --eval-episodes 1000 --load-dir simple_spread/m3ddpg-run-{2}_perturbation-{1}" ::: ${perturbation_factors[@]} ::: ${runs[@]}

    parallel -j 5 "python ../evaluate.py --eval-episodes 1000 --load-dir simple_spread_128_units/m3ddpg-run-{2}_perturbation-{1}" ::: ${perturbation_factors[@]} ::: ${runs[@]}

    parallel -j 5 "python ../evaluate.py --eval-episodes 1000 --load-dir simple_tag/{1}-vs-m3ddpg-run-{3}_perturbation-{2}" ::: ddpg m3ddpg ::: ${perturbation_factors[@]} ::: ${runs[@]}

    parallel -j 5 "python ../evaluate.py --eval-episodes 1000 --load-dir simple_tag_128_units/{1}-vs-m3ddpg-run-{3}_perturbation-{2}" ::: ddpg m3ddpg ::: ${perturbation_factors[@]} ::: ${runs[@]}
else
    echo "Unknown command"
fi