#!/bin/bash

pip install -e casmo10x10-master
pip install -e casmo6x6-master
pip install -e simulate3pwr-master

python3 - << EOF
import gym
print ("Checking env in Python3....")
if gym.make("casmo10x10:casmo10x10-v0"):
	print("10x10 env is installed correctly")
else:
	print("10x10 did not load in Python3")

if gym.make("casmo6x6:casmo6x6-v0"):
	print("6x6 env is installed correctly")
else:
	print("6x6 env did not load in Python3")
 
if gym.make("simulate3pwr:simulate3pwr-v0"):
	print("simulate3pwr env is installed correctly")
else:
	print("simulate3pwr env did not load in Python3")
 
EOF

