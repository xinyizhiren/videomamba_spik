1.处理数据集：VIT_4090\videomamba\video_sm\datasets\build.py(可以修改头文件引用的kinetics_sparse_et.py)
2.训练脚本：VIT_4090\videomamba\video_sm\exp\k400\videomamba_small\run_f16x224_1.sh（_1,_et,_g,_mst都是一些参数不同的文件可以改）
3.模型：\VIT_4090\videomamba\video_sm\models\videomamba.py(videomamba_spik.py是加了SNN的一版，要用可以直接复制一个新的)
4.主函数：VIT_4090\videomamba\video_sm\run_class_finetuning_et.py(证据融合版的，还有一些其他版本的，可以在训练脚本中改)
5.训练：VIT_4090\videomamba\video_sm\engines\engine_for_finetuning_et.py(可以设置缩略图大小数量，修改loss等等)
6.预训练模型：VIT_4090\videomamba\models\xxx.pth(可以videomamba的github上下载)



videomamba_spik.py版本的主要修改：
1.在 patch embedding 阶段加了 LIF神经元，把输入复制成多个时间步 timesteps，对所有时间步维度合并，得到脉冲化的 embedding 特征
2.在 Mamba Block 输出后加LIF 神经元脉冲化，且让 hidden states 先扩展成 timesteps 时序信号，经过 LIF 发放，再在时间维度上合并
3.当时的模型有24个Mamba Block，尝试只改前两层block，剩下的22层保留原来的
4.分类 head 没有脉冲化，保留了原本的全连接层。

之前存在的一些问题：
1.学习率高容易出现loss is nan
2.mamba层数设置多会导致准确率很低，lif也不能加太多层，具体多少还得尝试
3.训练过程很快，数据样本多一些效果会好很多
