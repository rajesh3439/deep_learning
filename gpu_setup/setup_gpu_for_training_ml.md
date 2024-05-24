This setup works for this system specification .
## Specs
- Primary OS: Windows 10
- WSL OS: Ubuntu 22.04
- Graphics Card: NVIDIA RTX A3000
- Python version : 3.10.12

## Mandatory Libraries
NVIDIA provides libraries to leverage the power of GPUs for training ML models. The following products are required to be installed
1) Cuda Toolkit
2) Cudnn
## Version Compatibility
[Tensorflow](https://www.tensorflow.org/install/source#gpu) and [Pytorch](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix) are the popular python packages for training Deep Learning models. It is very important to have the compatible versions of libraries and packages installed.

| Version           | Python version | cuDNN | CUDA |
| ----------------- | -------------- | ----- | ---- |
| tensorflow-2.14.0 | 3.9-3.11       | 8.7   | 11.8 |
| pytorch-2.1       | >=3.8, <=3.11  | 8.7   | 11.8 |
## Installation Steps
1) Download and Install graphics driver for Windows from [official site]([Official Drivers | NVIDIA](https://www.nvidia.com/Download/index.aspx?lang=en-us&_gl=1*dntj4n*_gcl_au*MjAzNDIyNTg4Ny4xNzE2NDMyNDcy)). Choose the driver for your specific graphics card. (Ensure you download the driver for Windows. Do not install graphics driver within WSL environment)
2) Install WSL (If not done before. Open PowerShell and use `wsl --install -d Ubuntu-22.04`)
3) **Install CUDA-Toolkit** :
	* Remove old GPG key:
			`sudo apt-key del 7fa2af80`
	* Execute below commands to install cuda-toolkit 
        ```shell
        wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
        sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
        wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-wsl-ubuntu-11-8-local_11.8.0-1_amd64.deb
        sudo dpkg -i cuda-repo-wsl-ubuntu-11-8-local_11.8.0-1_amd64.deb
        sudo cp /var/cuda-repo-wsl-ubuntu-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
        sudo apt-get update
        sudo apt-get -y install cuda	
        ```

	* Append the cuda-toolkit path to PATH environment variable. (Best practice is to add it in .bashrc file so it gets added to your environment variables automatically when you initialize terminal.)
	* Open .bashrc by running `vim ~/.bashrc` and add the below commands at the end of the file and save it
		```bash
		export CUDA_HOME=/usr/local/cuda
		export PATH=${CUDA_HOME}/bin:${PATH}
		export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH 
		```
	* Run `source ~/.bashrc` to update the environment variables in current session
	* To verify installation run `nvcc -V`. It shows cuda-toolkit version. 
	* **Reference** [Official guide for installing cuda-toolkit](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#:~:text=However%2C%20CUDA%20application%20development%20is,CUDA%20Toolkit%20for%20x86%20Linux), [Cuda-toolkit 11.8 package](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local)

5) **Install CUDNN** 
	- Install dependencies `sudo apt-get install zlib1g`
	- To download cudnn from NVIDIA, you need to setup an account and sign-in to [NVIDIA]([NVIDIA Developer](https://developer.nvidia.com/)).
	- Download the CUDNN [installer archive](https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-8.7.0.84_cuda11-archive.tar.xz), this download the file in your windows. Move it to linux filesystem.
	- Extract the archive 
		`tar xvf cudnn-linux-x86_64-8.7.0.84_cuda11-archive.tar.xz`
	- cd into the extracted folder
	- Copy libraries to cuda installation
		```shell
        sudo cp include/cudnn*.h /usr/local/cuda/include
        sudo cp -P lib/libcudnn* /usr/local/cuda/lib64                                          sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn* 
		```
	- Verify cudnn installation:
		- Install cudnn-samples 
			`sudo apt-get -y install libcudnn9-samples`
		- Copy samples to writable path for compiling
			`cp -r /usr/src/cudnn_samples_v9/ $HOME`
		- cd into mnist
			`cd  $HOME/cudnn_samples_v9/mnistCUDNN`
		- Compile
			`$make clean && make`
		- Run
			`./mnistCUDNN`
			If cudNN is properly installed and running you will see `Test Passed` message.
	- Reference : [cudNN Installation guide](https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-870/install-guide/index.html)

7) **Install Tensorflow** ![[Pasted image 20240523193309.png]]
	- Install tensor flow 2.14.0
			`pip install tensorflow[and-cuda]==2.14.0`
	- This installs all dependencies including keras, tensorRT all other dependencies. [Reference](https://www.tensorflow.org/install/pip#windows-wsl2)
	- Check tensor flow is able to detect GPU. Run the following command from terminal.
		`python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`

1) **Install Pytorch**![[Pasted image 20240523193240.png]]
	* Install Pytorch 2.1.0 
		`pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118`
	* [Reference to installation of PyTorch](https://pytorch.org/get-started/previous-versions/#linux-and-windows-11)
	* Check Pytorch is detecting the gpus
```python
>>> import torch
>>> torch.cuda.is_available()
True

>>> torch.cuda.device_count()
1

>>> torch.cuda.current_device()
0

>>> torch.cuda.device(0)
<torch.cuda.device at 0x7efce0b03be0>

>>> torch.cuda.get_device_name(0)
'GeForce GTX A3000'
```
[Ref: Basic Checks for Pytorch](https://stackoverflow.com/questions/48152674/how-do-i-check-if-pytorch-is-using-the-gpu)

## Helpful Links
- [Instructions for cuda 11.8 installation in Ubuntu 22.04](https://gist.github.com/Mahedi-61/2a2f1579d4271717d421065168ce6a73)