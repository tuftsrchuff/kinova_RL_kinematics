import matplotlib.pyplot as plt

timesteps = []
rewards = []
for i in range(0, 2):
    timesteps.append(i * 5000)

#PPO Rand both
f = open("./logs/rand_both_rewards.txt", "r")
for x in f:
  rewards.append(float(x))

f.close()
plt.plot(timesteps, rewards, label ='PPO')


plt.xlabel("Timesteps")
plt.ylabel("Mean reward per 100 episodes")
plt.title("Kinematics")
plt.legend()

plt.show()
