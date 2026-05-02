import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

class IGTEnv():
    def __init__(self,
                 mean_reward: np.ndarray, 
                 std_reward: np.ndarray, 
                 mean_loss: np.ndarray, 
                 std_loss: np.ndarray):
        self.num_arms = 4

        self.mean_reward = mean_reward
        self.std_reward = std_reward

        self.mean_loss = mean_loss
        self.std_loss = std_loss
        self.loss_tstamps = {0: np.array([0,1,0,1,0,1,0,1,0,1]),
                             1: np.array([0,0,0,0,1,0,0,0,0,0]),
                             2: np.array([0,1,0,1,0,1,0,1,0,1]),
                             3: np.array([0,0,0,0,1,0,0,0,0,0])}
        # shuffle each of the key in loss_tstamps
        for key in self.loss_tstamps:
            np.random.shuffle(self.loss_tstamps[key])

        # pick counter for each arm to give loss
        self.counts = np.zeros((self.num_arms))

        # asserts if the shapes donot match
        assert self.num_arms == self.mean_reward.shape[0] == self.std_reward.shape[0] == self.mean_loss.shape[0]== self.std_loss.shape[0], 'Invalid shape of mean_reward or std array'
        self.arms = dict(enumerate(zip(self.mean_reward, self.std_reward, self.mean_loss, self.std_loss)))

    def step(self, chosen_arm):
        arm_mean_rew, arm_dev_rew, arm_mean_loss, arm_dev_loss = self.arms[chosen_arm]

        # reward associated with the arm
        rew = np.random.normal(arm_mean_rew, arm_dev_rew)
        # loss associated with the arm
        loss = np.random.normal(arm_mean_loss, arm_dev_loss)
        loss = loss * self.loss_tstamps[chosen_arm][int(self.counts[chosen_arm])]
        rew = rew + loss

        self.counts[chosen_arm] +=1

        if self.counts[chosen_arm] == 10:
            # Once the arm is picked 10 times reset the counts and shuffle the array to change the loss times of the arm
            self.counts[chosen_arm] = 0
            np.random.shuffle(self.loss_tstamps[chosen_arm])
        return rew
    
    def reset(self):
        self.loss_tstamps = {0: np.array([0,1,0,1,0,1,0,1,0,1]),
                             1: np.array([0,0,0,0,1,0,0,0,0,0]),
                             2: np.array([0,1,0,1,0,1,0,1,0,1]),
                             3: np.array([0,0,0,0,1,0,0,0,0,0])}
        # shuffle each of the key in loss_tstamps
        for key in self.loss_tstamps:
            np.random.shuffle(self.loss_tstamps[key])

        # pick counter for each arm to give loss
        self.counts = np.zeros((self.num_arms))

        # asserts if the shapes donot match
        assert self.num_arms == self.mean_reward.shape[0] == self.std_reward.shape[0] == self.mean_loss.shape[0]== self.std_loss.shape[0], 'Invalid shape of mean_reward or std array'
        self.arms = dict(enumerate(zip(self.mean_reward, self.std_reward, self.mean_loss, self.std_loss)))