ENV_NAME := tvt
PYTHON_VERSION := 3.8
# conda run -n $(ENV_NAME) conda install -y pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
# conda run -n $(ENV_NAME) conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch-nightly -c nvidia -y

create_env:
	conda update -n base -c defaults conda -y
	conda create --name $(ENV_NAME) python=$(PYTHON_VERSION)  -y
	conda run -n $(ENV_NAME) conda install -y pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
	conda run -n $(ENV_NAME) pip install -r requirements.txt

# remove_envターゲット: condaの環境を削除
remove_env:
	conda env remove --name $(ENV_NAME)
