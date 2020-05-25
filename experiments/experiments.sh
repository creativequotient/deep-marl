#!/bin/bash

# Run experiments

cd simple_tag

bash simple_tag.sh

cd ../simple_push

bash simple_push_exp.sh

cd ../simple_spread

bash simple_spread_exp.sh

cd ../simple_spread_128_units

bash simple_spread_exp.sh

cd ../simple_speaker_listener

bash simple_speaker_listener.sh

cd ../simple_tag_128_units

 simple_tag_exp.sh

cd ../simple_adversary

bash simple_adversary_exp.sh

# Run analysis

parallel -j 5 "python ../evaluate.py --load-dir simple_tag/{1}-vs-{2}-run-{3} --eval-episodes 1000" ::: ddpg maddpg ::: ddpg maddpg ::: {1..10}

parallel -j 5 "python ../evaluate.py --load-dir simple_tag_128_units/{1}-vs-{2}-run-{3} --eval-episodes 1000" ::: ddpg maddpg ::: ddpg maddpg ::: {1..10}

parallel -j 5 "python ../evaluate.py --load-dir simple_push/{1}-vs-{2}-run-{3} --eval-episodes 1000" ::: ddpg maddpg ::: ddpg maddpg ::: {1..10}

parallel -j 5 "python ../evaluate.py --load-dir simple_spread/{1}-run-{2} --eval-episodes 1000" ::: ddpg maddpg ::: {1..10}

parallel -j 5 "python ../evaluate.py --load-dir simple_spread_128_units/{1}-run-{2} --eval-episodes 1000" ::: ddpg maddpg ::: {1..10}

parallel -j 5 "python ../evaluate.py --load-dir simple_speaker_listener/{1}-run-{2} --eval-episodes 1000" ::: ddpg maddpg ::: {1..10}

parallel -j 5 "python ../evaluate.py --load-dir simple_adversary/{1}-vs-{2}-run-{3} --eval-episodes 1000" ::: ddpg maddpg ::: ddpg maddpg ::: {1..10}
