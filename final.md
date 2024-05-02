---
layout: default
title: Final Report
---

## Video

<iframe width="1280" height="720" src="https://www.youtube.com/embed/4nK55kUSjhw" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Project Summary

Our agent's environment is set up as a survival course in Minecraft using the Malmo platform with a start point and end point. The agent learns to navigate this maze filled with magma, a healing item (golden apples), and a poisonous item (spider eyes). These food items are distributed throughout the maze to be collected and consumed. Because the agent only takes damage-over-time from touching magma instead of dying and resetting instantly, we hoped that we could train the agent to both safely navigate the maze and heal itself after making mistakes, all while avoiding poisoning itself. Our end goal would be to have the agent navigate the maze to the end point with full health.

We believed this challenge would be interesting because the benefit from food is not directly attached to the goal of finishing a maze. These food items do not do anything to increase the speed of the agent or give it more time to find the end of the maze. Rather, more or less time is _indirectly_ given to the agent as its health can act as a resource while exploring. This delayed action and reward is able to really test the limits of the algorithmic approach that we decide on.

## Approaches

### Baseline

As a sanity check, we started the agent off with just discrete actions, no food, and just a plain with an end block. We slowly added in items to its environment, eventually leading to magma blocks as "walls", stone for the ground, and glass and bedrock for item spawning.

We used the maze decorator included in Malmo in order to generate mazes. The maze was also sunken into the ground by one block; because the agent does not having jumping in his action space, this prevents it from going out of bounds. To generate the randomly placed items, pseudocode follows.

```
x = array of random int between 0 and length of maze
z = array of random int between 0 and width of maze
for i in range(chosen number of golden apples):
  add golden apple to field at x[i], z[i], and y=one block above surface
  add glass block to field at x[i], z[i], and y=surface level
  
x = array of random int between 0 and length of maze
z = array of random int between 0 and width of maze
for i in range(chosen number of spider eyes):
  add spider eye to field at x[i], z[i], and y=one block above surface
  add bedrock to field at x[i], z[i], and y=surface level
```

Because Malmo generates these added blocks after it creates a maze using a MazeDecorator, we also decided to always create a second diamond block on the field. This causes there to occasionally be two end blocks, but we decided this is better than occasionally having no end block at all for the agent. If there is no end block at all, it would be a bit discouraged from exploring if it doesn't immediately see a diamond block.

The final step was to move to a continous action space in order to enable the agent to eat the food that it is picking up. Eating food does not work with discrete actions in Malmo.

### Proposed Approach

We decided to initially tune our rewards to satisfactory performance while using unchanged default parameters for the PPO algorithm. The Proximal Policy Optimization (PPO) algorithm is described below.

![image](https://user-images.githubusercontent.com/50087239/144312978-29d792f6-f82e-4b00-968a-05257fa069bc.png)

PPO is an on-policy algorithm and is an easy method to implement and tune.

Xiao-Yang Liu, PhD from Columbia University and lead of the deep learning library ElegantRL, describes PPO as follows:

```
(PPO) follows a classic Actor-Critic framework with four components:
Initialization: initializes the related attributes and networks.
Exploring: explores transitions through the interaction between the Actor-network and the environment.
Computing: computes the related variables, such as the ratio term, exact reward, and estimate advantage.
Updating: updates the Actor and Critic networks based on the loss function and objective function.
```

We will be tuning the following parameters:
- Learning Rate: Self explanatory. This controls how fast the agent will learn during each episode.
- Gamma: A constant known as a discount factor. This allows us to tune how much future states are weighed while the agent is learning. Reaching the end block in the future should be _discounted_ compared to reaching the end block immediately.

We are added immediate view of the grid,agent's health bar, and the agent's hotbar items (not full inventory) to the observation space in our algorithm. This includes magma tiles, food items, and start and end blocks (if they are in the immediate view of the agent). Due to some issues with the Malmo representation of entities, we were unable to have the agent see the true items on the ground. Instead, we spawned them on unique blocks so that the agent could make associations that it would be able to pick up items by walking on certain blocks, in combination with its hotbar slots. Small negative rewards are given for every step the agent takes. Big positive rewards are given when reaching the end block (a diamond block). Our maze is located in a plain of stone. We tuned the rewards in order to encourage exploration while still encouraging the agent to take more efficient paths towards its destinations, and additionally we wanted to allow it to take some damage to create even shorter paths, but it must heal using golden apples in order to do so effectively. For our final run, we completely removed the passive healing the agent originally had as well. This makes it a requirement to eat golden apples to heal at all; original runs had the agent healing any time it wasn't taking damage.

### Alternative Approach, Pros and Cons

Initially, we were going to also test out Deep Q-Network (DQN), another reinforcement learning algorithm, as well, but we decided to funnel our efforts into having PPO's performance be improved due to limited time. During our research, we saw that PPO can converge faster, allowing for us to test quicker. Additionally, DQN usually relies on other additional algorithms in order to reach good performance, thus we were unsure if it was worth our time when we could spend more time increasing the current working performance we had.

Pseudocode for DQN is as follows.

![image](https://user-images.githubusercontent.com/50087239/144315600-127607a7-690f-4d06-a2c3-c87416bb80a6.png)

## Evaluation

### Qualitative

We watched the agent train live and also recorded some video to look back on in order to evaluate its qualitative performance.

There are more specific numbers and our methodology in measuring success in food choice over time in the quantitative section below. However, we noticed by just watching that the agent would initially eat food completely arbitrarily, essentially guessing 50/50 every time it decided to eat. By the end of our training, we noticed a clear preference towards golden apples. The majority of food it would eat would be golden apples, giving it more time to heal and thus more chances to explore.

After a lot of training, the agent does seem to be more careful with its movement than at the start. When the agent is first launched and training begins, its movement is extremely erratic as it simply tries out all of its available inputs. However, as time goes on, it does seem to make less extraneous inputs if it sees a diamond block (end goal for the maze). This is likely a result of its light punishment for every input it makes. However, given a different punishment value for inputs, this movement could have been even more refined or perhaps just stuck in a loop. Early iterations of the agent, especially before a time limit was added, would sometimes get stuck in an infinite loop as it decides to never explore and simply stay alive in one place. We found that reducing the punishment for each step significantly increased the rate at which the agent was willing to explore.

### Quantitative

As we trained, we used a function to log the rewards progress of the agent with every step, creating a graph for us to view over time. We hoped to see an upward trend in the graph as the agent trains.

Our first sanity check had an interesting issue. The agent learned that it was able to reset the map by going out of bounds without any punishment, and thus actually trained itself to immediately turn around and reset the map in order to avoid negative rewards from exploring and from dying. This is why the following rewards graph trends towards 0 while never reaching positive values:

![returns](https://user-images.githubusercontent.com/50087239/144315277-96df7dde-3a7f-4495-b04f-a6ad7be9ef32.png)

In order to fix this, we added walls to the maze. While the agent is still able to technically try to walk into the wall, we thought it would eventually learn that it is a waste of actions to try to walk into the wall, and thus train itself to stop over time anyways. This was simpler than reducing its action space when close to a wall, especially when thinking about the eventual goal of continuous actions. Unfortunately, a new issue was introduced over time. The agent learned to explore only a little bit to search for the end block. If it found it quickly, it would run to it, but if not, it would kill itself promptly. This is likely because we had an overly high negative reward on every step, discouraging exploration. However, its graph still trended upward with this goal in mind, with every spike representing one or more successful maze finishes.

![returns](https://user-images.githubusercontent.com/50087239/144316254-7622cc8a-9285-4e79-9794-0f80964ef7b5.png)

After allowing it to explore, we ran into one final issue before moving onto continuous actions. Now that the agent was encouraged to explore, it would sometimes simply get stuck in an infinite loop of walking in a circle if it could not find a diamond block relatively quickly. We theorized that this is because the agent learned that the diamond block has a high enough reward to search for, but its navigation skills were not adequate to explore the full maze safely. Thus, instead of taking a negative reward from dying, it simply chose to walk in circles infinitely. We solved this issue by simply adding a timeout of 30 seconds, with a negative reward equivalent to dying. The following is our final graph using discrete actions after these fixes were deployed.

![returns](https://user-images.githubusercontent.com/50087239/144316576-1aee542a-0b8a-4316-ae2c-eb2e49bd1c2c.png)

Finally, we moved on to continuous actions. Due to the way actions are implemented in Malmo, this was the first time we were able to actually allow the agent to eat. To accomodate for this, we also had to add hotbar item selection to its action space. Finally, because Minecraft agents are unable to eat food until their hunger bar is depleted, we gave the agent a starvation debuff upon every new maze. This has an additional effect of disallowing passive healing unless the agent eats a golden apple, which siginificantly increases the difficulty of reaching a diamond block. However, we actually had very good performance with continuous actions due to how aggressively the agent would explore. It would find end blocks very quickly, but it would not learn to consistently pick the correct foods. The following is a graph of the agent's rewards over time using continuous actions, including hotbar selection and eating, but without any changed parameters from the default of PPO.

![returns](https://user-images.githubusercontent.com/50087239/144317046-29499614-5be6-4d4e-a249-7461264b2dbb.png)

The final step we took was to run the agent with a Gamma value of 0.9 and a Learning Rate of 0.001. We arrived at these numbers after manually testing a couple of combinations of parameters for PPO. We would have liked to tune these numbers automatically, but we ran into some issues running that automation with the Ray library. The following was our final resulting rewards graph.

![returns](https://user-images.githubusercontent.com/50087239/144509820-3aaa3960-a52a-4815-8366-947b114508c0.png)

In the end it was difficult to observe much long term improvement for the agent's ability to navigate its environment, but it did have notable success in choosing between what food to eat. Without actually directly giving any rewards for foods eaten, we noticed that after several tens of thousands of trials, the agent is able to successfully pick the golden apple to eat when it requires healing most of the time. By taking random sample video and counting what food it eats manually, we were able to notice that golden apples are chosen about 85% of the time or higher, and the agent only chooses to eat spider eyes the remaining 15% of the time or lower. These random samples taken towards the end of the run are an accurate representation of the final result of the agent before it stopped training.

## References
https://towardsdatascience.com/elegantrl-mastering-the-ppo-algorithm-part-i-9f36bc47b791

https://towardsdatascience.com/proximal-policy-optimization-tutorial-part-2-2-gae-and-ppo-loss-fe1b3c5549e8
- We used these blog posts to learn about the higher level concepts of PPO and about the parameters we would eventually be tuning for it (gamma and learning rate).

https://medium.datadriveninvestor.com/which-reinforcement-learning-rl-algorithm-to-use-where-when-and-in-what-scenario-e3e7617fb0b1
- We used this blog post to learn about the difference between various reinforcement learning algorithms.
