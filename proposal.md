---
layout: default
title: Proposal
---

## Summary
Our agent's environment is set up as a survival course in Minecraft using the Malmo platform with a start point and end point. Agent learns how to navigate a maze which is covered with many magma tiles and food items. There are many food items distributed in the maze available for collection and consumption. Some our poisonous and some add health. We want our agent to be able to learn saturation values of various food items available to it. This is a hidden value that determines how long a food is able to keep a player full for, even beyond the regular displayed hunger bar. Our end goal for the agent is to be able to navigate the fastest path to the end in full health.


## AI/ML Algorithms
We are experimenting with two different Reinforcement algorithms. We have started with Proximal Policy Optimization(PPO) algorithm. PPO is an on-policy algorithm and is an easy method to implement and tune. We are planning to experiment with DQN which is based on Q network if we end up having some time in the end.


## Evaluation Plan
For quantitative evaluation, we will focus on time taken to get to the end of the maze. To measure this time quickly, we will apply a negative reward over time effect to the agent. We can take note of the accuracy of its choices to plot or view. The baseline would be to surpass a 50% success rate when picking between every pair of given food items. We hope to have the agent collect the right food item about 75% of the time and taking the shortest path to the end of the maze by the end of the training.

For qualitative evaluation, the sanity cases would be to give the agent a healing item and a poisonous item. It should quickly learn to choose the healing item. We will draw a box chart of the internals of our reinforcement learning algorithm with the necessary layers. Our moonshot case would be that our agent is able to pick the food items accurately every single time, and if we add a layer of complexity in which the agent must navigate a simple course while choosing the correct item, we would hope that the agent would stay on the path and not fall off in that case as well. Eventually we could increase the number of available food items to have the agent rank, which would increase the training time and the complexity of the problem greatly.
