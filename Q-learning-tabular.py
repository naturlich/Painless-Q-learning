# Code for a painless q-learning tutorial
# http://mnemstudio.org/path-finding-q-learning-tutorial.htm
# https://blog.csdn.net/itplus/article/details/9361915

import numpy as np

Q=np.matrix(np.zeros([6, 6]))
R=np.matrix([[-1, -1, -1, -1,  0,  -1], 
             [-1, -1, -1,  0, -1, 100], 
             [-1, -1, -1,  0, -1,  -1], 
             [-1,  0,  0, -1,  0,  -1], 
             [ 0, -1, -1,  0, -1, 100], 
             [-1,  0, -1, -1,  0, 100]])

num_episodes=100
gamma=0.8
epsilon=0.4

# Train
for episode in range(num_episodes):
	current = np.random.randint(0, 6)
	while current != 5:
		possible_actions=[]
		for action in range(6):
			#print("action: %d" % action)
			if R[current, action] >= 0:
				possible_actions.append(action)
		# Choose one available action
		choose = possible_actions[np.random.randint(0, len(possible_actions))]
		Q[current, choose] = R[current, choose] + gamma * np.max(Q[choose, :])

		# Update state
		current = choose
	print(Q)

# Test
num_episodes = 10

for episode in range(num_episodes):
	current = np.random.randint(0, 6)
	print("start from %d" % current)

	while current != 5:
		choose = np.argmax(Q[current, :])
		print("go to %d" % choose)

		current = choose
	print("-----------------------")
