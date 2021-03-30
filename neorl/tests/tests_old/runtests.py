# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 15:14:27 2020

@author: Majdi
"""

import os
import subprocess
import sys

if __name__=='__main__':
    #tests are running
    tid=['ga','ppo','sa', 'dqn', 'gridtune','randtune','gatune','bayestune']
    code=0
    neorl_path=os.path.join(sys.argv[2], 'neorl.py')
    passed=0
    for i in range(len(tid)):
        FNULL = open(os.devnull, 'w')
        code=subprocess.call(['python3', neorl_path, '-i', '{}.inp'.format(tid[i])], stdout=FNULL, stderr=subprocess.STDOUT, cwd='./src/tests')
        if code:
            print('--test {}/{}: {} was executed ---> FAIL X'.format(i+1,len(tid),tid[i]))
        else:
            print('--test {}/{}: {} was executed ---> PASS \u2713'.format(i+1,len(tid),tid[i]))
            passed+=1

    subprocess.call(['rm', '-rf', './src/tests/*_log'])
    print('--{}/{} tests were passed'.format(passed,len(tid)))
    