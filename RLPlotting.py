import matplotlib.pyplot as plt

timesteps = []
HERrewards = []
RandBothrewards = []
RandLocationrewards = []
RandArmrewards = []
RandResetrewards_mod = []
for i in range(0, 20):
    timesteps.append(i * 5000)

#HER
f = open("./rewards/HER_rewards.txt", "r")
for x in f:
  HERrewards.append(float(x))

f.close()
plt.plot(timesteps, HERrewards, label ='HER - Random Reset/Arm')

#Both
f = open("./rewards/rand_both_rewards.txt", "r")
for x in f:
  RandBothrewards.append(float(x))

f.close()
plt.plot(timesteps, RandBothrewards, label ='Random Reset/Arm')


#Rand Location, Const Reset
f = open("./rewards/rand_target_rewards.txt", "r")
for x in f:
  RandLocationrewards.append(float(x))

f.close()
plt.plot(timesteps, RandLocationrewards, label ='Random Location/Const Reset')

#Rand Reset, Const Location
f = open("./rewards/rand_arm_rewards.txt", "r")
for x in f:
  RandArmrewards.append(float(x))

f.close()
plt.plot(timesteps, RandArmrewards, label ='Random Reset/Const Location')


plt.xlabel("Timesteps")
plt.ylabel("Mean reward per 100 episodes")
plt.title("Kinova Reach Task")
plt.legend()

plt.show()
