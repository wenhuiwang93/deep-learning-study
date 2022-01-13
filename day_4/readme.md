Ubuntu 20.04 双系统 以及 Nvidia GPU 驱动安装。
系统安装：
工具：4G以上的U盘，ubuntu-20.04.3-desktop-amd64.iso ubuntu镜像文件，rufus系统制作工具。
参考网页：
https://blog.csdn.net/qq_45488453/article/details/107783752?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522164194901316780255240528%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=164194901316780255240528&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-3-107783752.first_rank_v2_pc_rank_v29&utm_term=ubuntu20.04%E5%AE%89%E8%A3%85%E6%95%99%E7%A8%8B&spm=1018.2226.3001.4187
注1：/boot 空间留300M左右· / 文件系统根目录空间留50G左右（CUDA，CUDNN， 以及后于python依赖库比较占空间）· /home 100G· linux swap空间 内存大小。
注2： 分区不足时，可以用gparted工具扩展，需usb启动ubuntu系统。

驱动安装：
参考网页：https://blog.csdn.net/ashome123/article/details/105822040
1.禁用nouveau

2. Software & Updates -> Additional Drivers
 选择 nvidia-driver-470(proprietary, tested) 自动安装。

3. 对应的cuda版本 11.4
  https://developer.nvidia.com/cuda-toolkit-archive
  选择 linux ubuntu 20.04 安装包
 
4. cuDnn v8.2.2
  https://developer.nvidia.com/cudnn
  选择 ubuntu 20.04 x86_64 runtime library 安装
  
校验：
Emotion-detection 
https://github.com/atulapra/Emotion-detection
GPU training OK。
