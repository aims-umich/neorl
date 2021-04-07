#!/bin/bash
echo "************************************************************************************"
echo "************************************************************************************"
echo "Welcome to NEORL Installer"
echo "************************************************************************************"
echo "************************************************************************************"
echo "Pick installation mode full or partial:"
echo "full: install from scratch including Anaconda/Python (step 1) and Numpy & Tensorflow (step 2) --> type full then press enter"
echo "partial: will skip steps 1 & 2, much faster option, and good when periodic updates on neorl-ONLY are to be installed --> type partial then press enter"
read mode
echo "$mode installation is selected"

if [ "$mode" != "full" ] && [ "$mode" != "partial" ]; then
  echo 1>&2 "--error: installation mode is invalid, either enter full or partial"
  exit 2
fi

here=$PWD
oneback="$(dirname "$here")"
echo "NEORL under=$here"
echo "Anaconda under=$oneback"

#*********************************************************************************************************
echo "------------------------------------------------------------"
echo "Step 1/6: Setup Python3/Anaconda3"
echo "------------------------------------------------------------"
if [ $mode == "full" ]; then
  wget --no-check-certificate https://repo.continuum.io/archive/Anaconda3-5.2.0-Linux-x86_64.sh
  bash Anaconda3-5.2.0-Linux-x86_64.sh -b -f -p $oneback/anaconda3
  rm Anaconda3-5.2.0-Linux-x86_64.sh
#  [[ $TRESHOLD =~ ^[0-9]+$ ]] || \
#   { echo "Fatal ERROR in step 1, Anaconda cannot be either downloaded or installed"; exit $ERRCODE; }
fi
P3PATH=$oneback/anaconda3/bin/python3
source $oneback/anaconda3/bin/activate base

#*********************************************************************************************************
echo "------------------------------------------------------------"
echo "Step 2/6: Setup Tensorflow and Numpy"
echo "------------------------------------------------------------"
if [ $mode == "full" ]; then
  echo "--installing numpy 1.17.4 ..."
  conda install -y numpy=1.17.4 
  echo "--installing tensorflow 1.13.1 ..."
  conda install -y tensorflow=1.13.1
#  [[ $TRESHOLD =~ ^[0-9]+$ ]] || \
#   { echo "Fatal ERROR in step 2, Numpy or Tensorflow cannot be updated"; exit $ERRCODE; }
fi

#*********************************************************************************************************
echo "------------------------------------------------------------"
echo "Step 3/6: Setup other packages"
echo "------------------------------------------------------------"
$P3PATH -m pip install -r requirements.txt

#*********************************************************************************************************
echo "------------------------------------------------------------"
echo "Step 4/6: Setup built-in enviroments"
echo "------------------------------------------------------------"
cd envs
chmod u+x install_env.sh 
./install_env.sh
cd ..

#*********************************************************************************************************
echo "------------------------------------------------------------"
echo "Step 5/6: Setup neorl path"
echo "------------------------------------------------------------"

echo "#-----------------------------" >> $HOME/.bashrc
echo "#------NEORL VARIABLES--------" >> $HOME/.bashrc
echo "export NEORLPATH=$here" >> $HOME/.bashrc
#string=':$PATH'
#echo "export PATH=$here/neorl${string}" >> $HOME/.bashrc
echo "alias neorl=\"$P3PATH $here/neorl.py -i\"" >> $HOME/.bashrc
echo "#-----------------------------" >> $HOME/.bashrc
source $HOME/.bashrc

#*********************************************************************************************************

#rm neorl
#%%%%%%%%%%%
# Template
#%%%%%%%%%%%

#echo "#!/bin/bash
#source ~/.bashrc
#------------------------------
# Main PATHS
#------------------------------
#ANACONDA3=$P3PATH
#NEORL=$here/neorl.py" >> neorl

#echo '#------------------------------
# Setup Enviroment Variables
#------------------------------
#export NEORLPATH=$NEORL
#export PATH="$NEORL:$PATH"
#export PATH="$ANACONDA3:$PATH"
#execute NEORL
#$ANACONDA3 $NEORLPATH' >> neorl

#%%%%%%%%%%%

#chmod +x neorl

#*********************************************************************************************************
echo "------------------------------------------------------------"
echo "Step 6/6: Run unit tests"
echo "------------------------------------------------------------"
#source ~/.bashrc
$P3PATH $here/src/tests/runtests.py -p $here
rm -rf $here/src/tests/*_log
rm -rf $here/src/tests/tunecases/
rm -rf $here/src/tests/*.csv
rm -rf $here/src/tests/*.txt

#*********************************************************************************************************
echo "----------------------------------------------------------------------------------------------"
echo "All steps are completed, type neorl on terminal to check installation (BETTER TO START A NEW TERMINAL)"
echo "----------------------------------------------------------------------------------------------"
$P3PATH $here/neorl.py
source ~/.bashrc
