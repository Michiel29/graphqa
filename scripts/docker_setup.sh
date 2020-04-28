# check if already instantiated
if [[ -d "/root/packages" ]]; then
    source activate self_inference
    cd /root/self_inference/self_inference
    exit
fi
​
# install basic programs
yes | apt-get install unzip
yes | apt-get install nano
yes | apt-get install wget
yes | apt install git
yes | apt install gcc
yes | apt-get install g++
​
# install anaconda
mkdir packages
cd packages
wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
chmod +x Anaconda3-2020.02-Linux-x86_64.sh
bash Anaconda3-2020.02-Linux-x86_64.sh -b -p /root/packages/anaconda3
echo 'PATH="/root/packages/anaconda3/bin:${PATH}"' >> ~/.bashrc
source ~/.bashrc
conda init
cd /root/
​
​
# download repo and data
mkdir self_inference
cd self_inference
git clone https://Michiel29Machine:uWnSK*hsG6VU69AW@github.com/Michiel29/self_inference.git
​
mkdir data
mkdir save
​
# wget datapath
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1YUOrLeogvUPls3OJW0a0lvop0Jn_FE_R' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1YUOrLeogvUPls3OJW0a0lvop0Jn_FE_R" -O data.zip && rm -rf /tmp/cookies.txt
unzip data
​
​
# create env and install deps
yes | conda create --name self_inference
source activate self_inference
yes | conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
​
pip install fastBPE
pip install sacremoses
pip install requests
pip install commentjson
pip install pyarrow
pip install neptune-client
pip install psutil
pip install sklearn
​
cd /root/packages
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--deprecated_fused_adam" --global-option="--xentropy" --global-option="--fast_multihead_attn" ./
​
cd /root/packages
git clone https://github.com/pytorch/fairseq
cd fairseq
git checkout b5dad3b7e02d66dc98d4707bc8aeacf95618ccd7
pip install --editable .
​
cd /root/self_inference/self_inference
python setup.py build_ext --inplace