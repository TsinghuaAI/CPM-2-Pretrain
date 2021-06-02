# CPM-2 Pre-Train

Pre-train CPM-2 
**此分支为110亿非 MoE 模型的预训练代码，MoE 模型的预训练代码请切换到 moe 分支**

CPM-2技术报告请参考[link](https://github.com/TsinghuaAI/tsinghuaai.github.io/blob/main/CPM-2.pdf)。

## 0 模型下载

请在智源资源[下载页面](https://resource.wudaoai.cn/home?ind=2&name=WuDao%20WenYuan&id=1394901846484627456)进行申请，文件介绍如下：

| 文件名 | 描述 | 参数大小 |
| :-----| ----: | :----: |
| 100000.tar | 纯中文模型 | 110亿 |
| 36000.tar | 中英文双语模型 | 110亿 |
| 300000.tar | 中英文MoE模型 | 1980亿 |

## 1 安装
可以直接拉取我们提供的 Docker 环境：

```[bash]
docker pull gyxthu17/cpm-2:1.0
```

## 2 数据
`scripts/gen_data.sh` 中给出了生成数据文件的脚本示例。该脚本将一个多行的纯文本文件（一个 document 一行）转化为二进制文件（会输出三个 .bin 和三个 .idx 文件），方便模型读取。

## 3 训练
首先需要将 `WORKING_DIR` 变量换成 CPM-2 目录的所在路径。调整 `NUM_WORKERS` 和 `NUM_GPUS_PER_WORKER` 指定机器数量与每台机器的 GPU 设备数量。修改 `${WORKING_DIR}/src/configs/host_files/hostfile-cpm2` 文件将其中的主机名称替换成每台机器的 IP 地址或者和 IP 地址相关联的主机名称。

运行命令：
```[bash]
cd src
bash scripts/pretrain_enc_dec.sh
```

## 4 引用
如果您使用了我们的代码，请您引用下面的文章。
```
@article{cpm-v2,
  title={CPM-2: Large-scale Cost-efficient Pre-trained Language Models},
  author={Zhang, Zhengyan and Gu, Yuxian and Han, Xu and Chen, Shengqi and Xiao, Chaojun and Sun, Zhenbo and Yao, Yuan and Qi, Fanchao and Guan, Jian and Ke, Pei and Cai, Yanzheng and Zeng, Guoyang and Tan, Zhixing and Liu, Zhiyuan and Huang, Minlie and Han, Wentao and Liu, Yang and Zhu, Xiaoyan and Sun, Maosong},
  year={2021}
}
```
