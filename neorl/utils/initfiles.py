#    This file is part of NEORL.

#    Copyright (c) 2021 Exelon Corporation and MIT Nuclear Science and Engineering
#    NEORL is free software: you can redistribute it and/or modify
#    it under the terms of the MIT LICENSE

#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#    SOFTWARE.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 09:33:45 2020

@author: majdi
"""
import numpy as np
import pandas as pd
import os
import shutil

def initfiles(methods, nx, ny, inp_headers, out_headers, log_dir, logo):
    
    
    if len(out_headers) >= 1 and out_headers[0] != 'y':
        assert ny == len(out_headers), 'number of outputs assigned in ysize ({}) is not equal to ynames ({})'.format(ny, len(out_headers))
    if len(inp_headers) >= 1 and inp_headers[0] != 'x':
        assert nx == len(inp_headers), 'number of inputs assigned in xsize_plot ({}) is not equal to xnames ({})'.format(nx, len(inp_headers))
    
    old_log_dir='old_'+log_dir
    # check if the log directory exists, move to old and create a new log
    if os.path.exists(log_dir) and os.path.exists(old_log_dir):
        try:
            os.system('rm -Rf {}'.format(old_log_dir))
        except:
            shutil.rmtree('./'+old_log_dir)
        os.rename(log_dir,old_log_dir)
        os.makedirs(log_dir)
    elif os.path.exists(log_dir): 
        os.rename(log_dir,old_log_dir)
        os.makedirs(log_dir)         
    else:
        os.makedirs(log_dir)
        
    inp_names=['caseid', 'reward'] 
    if len(inp_headers) == 1 and inp_headers[0]=='x':
        [inp_names.append('x'+str(i)) for i in range(1,nx+1)]
    else:
        [inp_names.append(i) for i in inp_headers]
        
    
    out_names=['caseid', 'reward']
    if len(out_headers) == 1 and out_headers[0]=='y':
        [out_names.append('y'+str(i)) for i in range(1,ny+1)]
    else:
        [out_names.append(i) for i in out_headers]
        
    for method in methods:
        
        with open (log_dir+method+'_inp.csv','w') as fin:
            for i in range(len(inp_names)):
                if i==len(inp_names)-1:
                    fin.write(inp_names[i] +'\n')
                else:
                    fin.write(inp_names[i]+',')
                                
        with open (log_dir+method+'_out.csv','w') as fin:
            for i in range(len(out_names)):
                if i==len(out_names)-1:
                    fin.write(out_names[i] + '\n')
                else:
                    fin.write(out_names[i]+',')
                    
        with open (log_dir+method+'_summary.txt','w') as fin:
            fin.write('---------------------------------------------------\n')
            fin.write('Summary file for the {} method \n'.format(method))
            fin.write('---------------------------------------------------\n')
            #fin.write(logo)
            
    print('--debug: All logging files are created')