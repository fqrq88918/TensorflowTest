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

注意：如果找不到对应的framework，去集成环境下找，比如用Homebrew安装的python就不在上述文件夹下

4/ 通过tensorflow的主页 https://www.tensorflow.org/install/install_mac#the_url_of_the_tensorflow_python_package，下载相应版本的安装whl文件

5/ 安装Anaconda
通过Anaconda官网下载 https://www.anaconda.com/download/#macos

使用图形界面安装python3.5 
地址在你Anaconda安装目录下/envs/你的python目录命名

命令 激活python版本 source activate python35。退出python环境 source deactivate python35

6/ 使用pip安装tensorflow
先进入python3.5环境（因为之前3.6安装出错了）
###  注意，这里的3.6环境 是homebrew的 不好控制 最后我们还是使用Anaconda的python3.6
source activate python35
sudo pip install --upgrade 你的whl安装文件的地址

如果提示权限问题 输入以下命令
sudo chown root /Users/hfcb/Library/Caches/pip/http
sudo chown root /Users/hfcb/Library/Caches/pip

7/ 安装python的IDE 暂时选择pycharm

