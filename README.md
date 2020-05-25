# Deep Multi-Agent Reinforcement Learning
AY2020/21 Sem 1/2 CP3209 UROP in Computing Project with Dr Jing Wei, IHPC.

# To-do list

## High Priority
- [x] Get models working for speaker_listener, followed by the rest of the scenarios
- [x] Add discrete action space output option via Gumbel-Softmax reparameterization trick
- [x] Move noise parameter to inside the agent class
- [x] Add support for individual good/bad agent policies
- [ ] Implement M3DDPG algorithm
- [ ] Implement policy estimation and esembling for MADDPG

## Medium Priority
- [ ] Add support for `MultiBoxDiscrete` action space
- [x] Add individual agent reward tracking
- [x] Experiment with additional normalization layers
- [ ] Experiment with separate actor/critic networks

## Low Priority
- [x] Re-factorization of code into packages
- [ ] Modify MPE code to provide benchmark statistics
- [x] Document code
- [ ] Add ability to set random seed
- [ ] Add printout for model and experimental parameters before code execution
