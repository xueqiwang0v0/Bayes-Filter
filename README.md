# Bayes-Filter

In this problem we are given a robot operating in a 2D gridworld. 
Every cell in the gridworld is characterized by a color (0 or 1). 
The robot is equipped with a noisy odometer and a noisy color sensor. 

Given a stream of actions and corresponding observations, a Bayes filter is applied 
to keep track of the robot’s current position.

## Models

### Sensor Model
* z = true observation    with probability 0.9
* z = false observation   with probability 0.1


### Action Model
* x[t+1] = x[t] + u[t]    with probability 0.9
* x[t+1] = x[t]           with probability 0.1

When the robot is at the edge of the gridworld and is tasked with executing an action
that would take it outside the boundaries of the gridworld, the robot remains in the same
state with p = 1. Start with a uniform prior on all states. For example if you have a world
with 4 states (s1,s2,s3,s4) then P(x = s1) = P(x = s2) = P(x = s3) = P(x = s4) = 0.25.

## Files
The starter.npz file contains a binary color-map, a sequence of actions, a sequence of
observations, and a sequence of the correct belief states.

A bayes filter is implemented in the histogram_filter.py.

The example_test.py can be used to test the filter.
