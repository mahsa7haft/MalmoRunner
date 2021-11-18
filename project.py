# Rllib docs: https://docs.ray.io/en/latest/rllib.html
# Malmo python examples: https://canvas.eee.uci.edu/courses/34142/pages/python-examples-malmo-functionality
# Malmo XML docs:https://microsoft.github.io/malmo/0.30.0/Schemas/Mission.html
# Harvesting Food
try:
    from malmo import MalmoPython
except:
    import MalmoPython

import sys
import time
import json
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint
import random
import gym, ray
from gym.spaces import Discrete, Box
from ray.rllib.agents import ppo


class DiamondCollector(gym.Env):

    def __init__(self, env_config):  
        # Static Parameters
        self.size = 50
        self.reward_density = .2
        self.penalty_density = .7
        self.obs_size = 5
        self.max_episode_steps = 100
        self.log_frequency = 10
        self.action_dict = {
            0: 'move 1',  # Move one block forward
            1: 'turn 1',  # Turn 90 degrees to the right
            2: 'turn -1',  # Turn 90 degrees to the left
            3: 'use 1'  # eat fruit
        }

        # Rllib Parameters
        self.action_space = Discrete(len(self.action_dict))
        #self.action_space = Box(-1, 1, shape=(3,), dtype=np.float32)        
        self.observation_space = Box(0, 1, shape=(101, ), dtype=np.float32)

        # Malmo Parameters
        self.agent_host = MalmoPython.AgentHost()
        try:
            self.agent_host.parse( sys.argv )
        except RuntimeError as e:
            print('ERROR:', e)
            print(self.agent_host.getUsage())
            exit(1)

        # DiamondCollector Parameters
        self.obs = None
        self.allow_break_action = False
        self.episode_step = 0
        self.episode_return = 0
        self.returns = []
        self.steps = []

    def reset(self):
        """
        Resets the environment for the next episode.

        Returns
            observation: <np.array> flattened initial obseravtion
        """
        # Reset Malmo
        world_state = self.init_malmo()

        # Reset Variables
        self.returns.append(self.episode_return)
        current_step = self.steps[-1] if len(self.steps) > 0 else 0
        self.steps.append(current_step + self.episode_step)
        self.episode_return = 0
        self.episode_step = 0

        # Log
        if len(self.returns) > self.log_frequency + 1 and \
            len(self.returns) % self.log_frequency == 0:
            self.log_returns()

        # Get Observation
        self.obs, self.allow_break_action = self.get_observation(world_state)

        return self.obs

    def step(self, action):
        """
        Take an action in the environment and return the results.

        Args
            action: #<int> index of the action to take#
            new action: <Box> 

        Returns
            observation: <np.array> flattened array of obseravtion
            reward: <int> reward from taking action
            done: <bool> indicates terminal state
            info: <dict> dictionary of extra information
        """
        # Get Action

        command = self.action_dict[action]
        if command != 'use 1' or self.allow_break_action:
            self.agent_host.sendCommand(command)
            time.sleep(.1)
            self.episode_step += 1
        
        # Get Observation
        world_state = self.agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:", error.text)
        self.obs, self.allow_break_action = self.get_observation(world_state)

        

        # Get Done
        done = not world_state.is_mission_running 
        # Get Reward
        reward = 0
        for r in world_state.rewards:
            reward += r.getValue()
        #to acount for time
        reward -= 0.1
        self.episode_return += reward
    
        return self.obs, reward, done, dict()


    def get_mission_xml(self):
        healthyPool = ['golden_apple', 'golden_carrot', 'cooked_beef']
        poisonPool = ['spider_eye', 'poisonous_potato', 'chicken']

        num = int(10)
        x = randint(0,20,size=int(num))
        z = randint(0,20,size=int(num))
        addXml = ""
        for i in range(num):
            addXml +="<DrawItem x='{}' y='3' z='{}' type='golden_apple'/> ".format(x[i],z[i])
            addXml +="<DrawBlock x='{}' y='2' z='{}' type='glass'/> ".format(x[i],z[i])
        num = int(10)
        x = randint(0,20,size=int(num))
        z = randint(0,20,size=int(num))
        for i in range(num):
            addXml +="<DrawItem x='{}' y='3' z='{}' type='chicken'/> ".format(x[i],z[i])
            addXml +="<DrawBlock x='{}' y='2' z='{}' type='bedrock'/> ".format(x[i],z[i])
        
        
        return '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
                <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">

                    <About>
                        <Summary>Diamond Collector</Summary>
                    </About>

                    <ServerSection>
                        <ServerInitialConditions>
                            <Time>
                                <StartTime>12000</StartTime>
                                <AllowPassageOfTime>false</AllowPassageOfTime>
                            </Time>
                            <Weather>clear</Weather>
                        </ServerInitialConditions>
                        <ServerHandlers>
                            <FlatWorldGenerator generatorString="3;7,2;1;"/>
                            <MazeDecorator>
                                <SizeAndPosition length="20" width="20" xOrigin="0" yOrigin="1" zOrigin="0" height="0"/>
                                <GapProbability variance="0.2">0.5</GapProbability>
                                <Seed>random</Seed>
                                <MaterialSeed>random</MaterialSeed>
                                <AllowDiagonalMovement>false</AllowDiagonalMovement>
                                <StartBlock fixedToEdge="true" type="redstone_block" height="1"/>
                                <EndBlock fixedToEdge="true" type="diamond_block" height="1"/>
                                <PathBlock type="stone" height="1"/>
                                <FloorBlock type="magma" height="1"/>
                                <GapBlock type="magma" height="1"/>
                                <AddQuitProducer description="finished maze"/>
                                <AddNavigationObservations/>
                            </MazeDecorator>
                            <DrawingDecorator>
                                <DrawCuboid type="air" x1='0' y1='3' z1='0' x2='20' y2='3' z2='20' />''' + \
                                addXml +\
                                '''
                            </DrawingDecorator>
                            <ServerQuitWhenAnyAgentFinishes/>
                        </ServerHandlers>
                    </ServerSection>
                    <AgentSection mode="Survival">
                        <Name>CS175DiamondCollector</Name>
                        <AgentStart>
                            <Placement x="0.5" y="2" z="0.5" pitch="45" yaw="0"/>
                            <Inventory>
                                <InventoryItem slot="0" type="diamond_pickaxe"/>
                            </Inventory>
                        </AgentStart>
                        <AgentHandlers>
                            <RewardForTouchingBlockType>
                               <Block type="magma" reward="-1"/>
                               <Block type="diamond_block" reward="20"/>
                               <Block type="grass" reward="-10"/>
                            </RewardForTouchingBlockType>
                            <DiscreteMovementCommands/>
                            <ObservationFromFullStats/>
                            <ObservationFromRay/>
                            <ObservationFromNearbyEntities>
                                <Range name="Entities" xrange="10" yrange="1" zrange="10"/>
                            </ObservationFromNearbyEntities>
                            <ObservationFromGrid>
                                <Grid name="floorAll">
                                    <min x="-'''+str(int(self.obs_size/2))+'''" y="-1" z="-'''+str(int(self.obs_size/2))+'''"/>
                                    <max x="'''+str(int(self.obs_size/2))+'''" y="-1" z="'''+str(int(self.obs_size/2))+'''"/>
                                </Grid>
                            </ObservationFromGrid>
                            <AgentQuitFromTouchingBlockType>
                                <Block type="grass" />
                            </AgentQuitFromTouchingBlockType>
                        </AgentHandlers>
                    </AgentSection>
                </Mission>'''
    def init_malmo(self):
        """
        Initialize new malmo mission.
        """
        my_mission = MalmoPython.MissionSpec(self.get_mission_xml(), True)
        my_mission_record = MalmoPython.MissionRecordSpec()
        my_mission.requestVideo(800, 500)
        my_mission.setViewpoint(1)

        max_retries = 3
        my_clients = MalmoPython.ClientPool()
        my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000)) # add Minecraft machines here as available

        for retry in range(max_retries):
            try:
                self.agent_host.startMission( my_mission, my_clients, my_mission_record, 0, 'DiamondCollector' )
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission:", e)
                    exit(1)
                else:
                    time.sleep(2)

        world_state = self.agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()
            for error in world_state.errors:
                print("\nError:", error.text)

        return world_state

    def get_observation(self, world_state):
        """
        Use the agent observation API to get a flattened 2 x 5 x 5 grid around the agent. 
        The agent is in the center square facing up.

        Args
            world_state: <object> current agent world state

        Returns
            observation: <np.array> the state observation
            allow_break_action: <bool> whether the agent is facing a diamond
        """
        obs = np.zeros((4 * self.obs_size * self.obs_size, ))
        allow_break_action = False

        while world_state.is_mission_running:
            time.sleep(0.5)
            world_state = self.agent_host.getWorldState()
            if len(world_state.errors) > 0:
                raise AssertionError('Could not load grid.')

            if world_state.number_of_observations_since_last_state > 0:
                # First we get the json from the observation API
                msg = world_state.observations[-1].text
                observations = json.loads(msg)

                # Get observation
                grid = observations['floorAll']
                items = observations['Entities']
                Life = observations['Life']


                #Normalize life score
                life_score = Life / 20

                #print("this is grid"+ str(grid))
                #print("This is life"+str(Life))

                #print("This is items:"+ str(items))

                i = 0

                for x in grid:
                    obs[i] = x == 'glass'
                    i+=1

                for x in grid:
                    obs[i] = x == 'magma'
                    i+=1

                for x in grid:
                    obs[i] = x == 'bedrock'
                    i+=1

                for x in grid:
                    obs[i] = x =='diamond_block'
                    i+=1
            
                
                # Rotate observation with orientation of agent
                obs = obs.reshape((4, self.obs_size, self.obs_size))
                yaw = observations['Yaw']
                if yaw >= 225 and yaw < 315:
                    obs = np.rot90(obs, k=1, axes=(1, 2))
                elif yaw >= 315 or yaw < 45:
                    obs = np.rot90(obs, k=2, axes=(1, 2))
                elif yaw >= 45 and yaw < 135:
                    obs = np.rot90(obs, k=3, axes=(1, 2))

                obs = obs.flatten()
                obs = np.append(obs, life_score)

                #allow_break_action = observations['LineOfSight']['type'] == 'golden_apple'
                #allow_break_action = observations['LineOfSight']['type'] == 'chicken'
                
                break
        if world_state.is_mission_running != True:
            obs = np.zeros((4 * self.obs_size * self.obs_size+1, ))

        return obs, allow_break_action
    
    def log_returns(self):
        """
        Log the current returns as a graph and text file

        Args:
            steps (list): list of global steps after each episode
            returns (list): list of ttal return of each episode
        """
        box = np.ones(self.log_frequency) / self.log_frequency
        returns_smooth = np.convolve(self.returns[1:], box, mode='same')
        plt.clf()
        plt.plot(self.steps[1:], returns_smooth)
        plt.title('DiamondCollector')
        plt.ylabel('Return')
        plt.xlabel('Steps')
        plt.savefig('returns.png')

        with open('returns.txt', 'w') as f:
            for step, value in zip(self.steps[1:], self.returns[1:]):
                f.write("{}\t{}\n".format(step, value)) 


if __name__ == '__main__':
    ray.init()
    trainer = ppo.PPOTrainer(env=DiamondCollector, config={
        'env_config': {},           # No environment parameters to configure
        'framework': 'torch',       # Use pyotrch instead of tensorflow
        'num_gpus': 0,              # We aren't using GPUs
        'num_workers': 0            # We aren't using parallelism
    })

    while True:
        print(trainer.train())
