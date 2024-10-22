# TODO List for StrideZero

## High Priority
- [ ] **Write the TRPO algorithm**: Implement the core components of the TRPO algorithm.
  - [ ] Implement trajectory collection
  - [ ] Implement policy optimization using KL divergence constraint
  - [ ] Implement value function update
  - [ ] Test and debug the algorithm

- [ ] **Test TRPO on easy environment (Continuous CartPole)**
  - [ ] Set up Continuous CartPole environment
  - [ ] Track performance (e.g., reward over episodes, convergence rate)

- [ ] **Test TRPO on medium difficulty environment (Bipedal Humanoid 2D)**
  - [ ] Set up Bipedal Humanoid 2D environment
  - [ ] Track performance (e.g., reward, stability of walking)

- [ ] **Test TRPO on hard environment (3D Humanoid)**
  - [ ] Set up 3D Humanoid environment
  - [ ] Track performance (e.g., reward, stability, convergence time)

- [ ] **Figure out performance tracking**
  - [ ] Track episode reward, running average
  - [ ] Implement logging for key metrics (reward, loss, KL divergence, ec.)

## Medium Priority
- [ ] **Write a grid search for hyperparameter tuning**
  - [ ] Tune learning rate, KL divergence threshold, batch size, etc.
  - [ ] Set up automated grid search over hyperparameters
  - [ ] Store results and analyze the best configuration

- [ ] **Use JAX to parallelize environments**
  - [ ] Modify environment code to run environments in parallel using JAX
  - [ ] Collect more training data efficiently with parallelized simulations

## Low Priority
- [ ] **Write tests for the replay buffer**
  - [ ] Ensure the replay buffer is correctly storing and sampling transitions
  - [ ] Test for edge cases (e.g., empty buffer, buffer overflow)

- [ ] **Improve logging and monitoring**
  - [ ] Add detailed logs for each major step in the TRPO algorithm
  - [ ] Set up TensorBoard or other tools to monitor performance in real-time

- [ ] **Write documentation for TRPO implementation**
  - [ ] Add detailed comments to the TRPO code
  - [ ] Document the architecture and design choices

