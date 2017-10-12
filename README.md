# TensorflowTest
TensorflowTest for LM

Tensorflow 安装 MAC
1/ 安装pip 
sudo easy_install pip
sudo easy_install --upgrade six

2/ 安装 Homebrew
macOS 缺失的软件包管理器
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

3/ 用Homebrew安装python3（最好用Anaconda安装python）
brew install python3

** 当前最新版本为3.6，但是安装tensorflow时出错，故删除再用Anaconda安装3.5版本
删除Python 3.6 framework

sudo rm -rf /Library/Frameworks/Python.framework/Versions/3.6

删除Python 3.6 应用目录

sudo rm -rf “/Applications/Python 3.6”

删除/usr/local/bin 目录下指向的Python3.6的连接

cd /usr/local/bin/ 
ls -l /usr/local/bin | grep ‘../Library/Frameworks/Python.framework/Versions/3.6’ | awk ‘{print $9}’ | tr -d @ | xargs rm

4/ 通过tensorflow的主页 https://www.tensorflow.org/install/install_mac#the_url_of_the_tensorflow_python_package，下载相应版本的安装whl文件

sudo chown root /Users/hfcb/Library/Caches/pip/http
sudo chown root /Users/hfcb/Library/Caches/pip
5/ 安装Anaconda
  



