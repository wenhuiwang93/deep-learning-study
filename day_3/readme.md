# 环境：
## 操作系统
   - ubuntu 20.04 
## kernel
   - 5.11.0-27-generic
## Nvidia GPU
   - Driver Version: 470.141.03
   - CUDA Version: 11.4 
# Tensorflow下的EEG分类
## 依赖环境
   - python == 3.8.10
   - tensorflow==2.5.2
   - scikit-learn==1.0.2
   - scipy==1.7.3
   - numpy==1.19.2
## 预处理(生成EEG图像)
### 坐标转换
```
for e in locs_3d:
  locs_2d.append(azim_proj(e))
```
三维坐标转二维投影坐标
### 图像插值
```
    for i in range(nSamples):
        for c in range(n_colors):
            temp_interp[c][i, :, :] = griddata(locs, feat_array_temp[c][i, :], (grid_x, grid_y),    # cubic
                                    method='cubic', fill_value=np.nan)
```
## 搭建深度学习网络（LSTM）
```
def build_convpool_lstm(input_image, nb_classes, grad_clip=110, image_size=32, n_colors=3, 
        n_timewin=7, dropout_rate=0.5, num_units=128, batch_size=32, name='CNN_LSTM', train=True, reuse=False):
    with tf.compat.v1.name_scope(name):
        with tf.compat.v1.name_scope('Parallel_CNNs'):
            convnets = []
            # Build 7 parallel CNNs with shared weights
            for i in range(n_timewin):
                if i==0:
                    convnet = build_cnn(input_image[i],image_size=image_size,n_colors=n_colors, reuse=reuse)
                else:
                    convnet = build_cnn(input_image[i],image_size=image_size,n_colors=n_colors, reuse=True)
                convnets.append(convnet)
            convnets = tf.stack(convnets)
            convnets = tf.transpose(a=convnets, perm=[1,0,2,3,4]) # 调换轴 shape: (nSamples, n_timewin, 4, 4, 128)

        with tf.compat.v1.variable_scope('LSTM_layer'):
            # (nSamples, n_timewin, 4, 4, 128) ==>  (nSamples, n_timewin, 4*4*128)
            convnets = tf.reshape(convnets, shape=[-1, n_timewin, 4*4*128], name='Reshape_for_lstm')
            #lstm cell inputs:[batchs, time_steps, 4*4*128]
            with tf.compat.v1.variable_scope('LSTM_Cell'):
                lstm_cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(num_units=num_units, forget_bias=1.0, state_is_tuple=True)
                outputs, final_state = tf.compat.v1.nn.dynamic_rnn(lstm_cell, convnets, dtype=tf.float32, time_major=False)
                # outputs.shape is (batch_size, time_steps, num_units)
                outputs = tf.transpose(a=outputs, perm=[1,0,2])        # (time_steps, batch_size, num_units)
                outputs = outputs[-1]

        with tf.compat.v1.variable_scope('Output_layers'):
            h_fc1_drop1 = tf.compat.v1.layers.dropout(outputs, rate=dropout_rate, training=train, name='dropout_1')
            h_fc1 = tf.compat.v1.layers.dense(h_fc1_drop1, 256, activation=tf.nn.relu, name='fc_relu_256')
            h_fc1_drop2 = tf.compat.v1.layers.dropout(h_fc1, rate=dropout_rate, training=train, name='dropout_2')
            prediction = tf.compat.v1.layers.dense(h_fc1_drop2, nb_classes, name='fc_softmax')

    return prediction
```
## Train
python train.py
```
--------------------------------------------------
Epoch 55 of 80 took 6.534s
Train   Epoch [55/80]  train_Loss: 0.7057       train_Acc: 84.22
Val     Epoch [55/80]  val_Loss: 0.5256 val_Acc: 92.43
Test    Epoch [55/80]  test_Loss: 1.0254        test_Acc: 52.97
--------------------------------------------------
```

# Pytorch下的EEG分类
## 环境配置
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
使用镜像站的场合，请删除“-c pytorch”
使用conda导入环境：conda env create -f Pytorch_EEG.yml
## 预处理
和Tensorflow一致
## 搭建深度学习网络（LSTM）
```
class LSTM(nn.Module):
    def __init__(self, input_image=torch.zeros(1, 7, 3, 32, 32), kernel=(3, 3), stride=1, padding=1, max_kernel=(2, 2),
                 n_classes=4, n_units=128):
        super(LSTM, self).__init__()

        n_window = input_image.shape[1]
        n_channel = input_image.shape[2]

        self.conv1 = nn.Conv2d(n_channel, 32, kernel, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(32, 32, kernel, stride=stride, padding=padding)
        self.conv3 = nn.Conv2d(32, 32, kernel, stride=stride, padding=padding)
        self.conv4 = nn.Conv2d(32, 32, kernel, stride=stride, padding=padding)
        self.pool1 = nn.MaxPool2d(max_kernel)
        self.conv5 = nn.Conv2d(32, 64, kernel, stride=stride, padding=padding)
        self.conv6 = nn.Conv2d(64, 64, kernel, stride=stride, padding=padding)
        self.conv7 = nn.Conv2d(64, 128, kernel, stride=stride, padding=padding)

        # LSTM Layer
        self.rnn = nn.RNN(4 * 4 * 128, n_units, n_window)
        self.rnn_out = torch.zeros(2, 7, 128)

        self.pool = nn.MaxPool2d((n_window, 1))
        self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear(896, n_classes)
        self.max = nn.LogSoftmax()

    def forward(self, x):
        if x.get_device() == 0:
            tmp = torch.zeros(x.shape[0], x.shape[1], 128, 4, 4).cuda()
        else:
            tmp = torch.zeros(x.shape[0], x.shape[1], 128, 4, 4).cpu()
        for i in range(7):
            img = x[:, i]
            img = F.relu(self.conv1(img))
            img = F.relu(self.conv2(img))
            img = F.relu(self.conv3(img))
            img = F.relu(self.conv4(img))
            img = self.pool1(img)
            img = F.relu(self.conv5(img))
            img = F.relu(self.conv6(img))
            img = self.pool1(img)
            img = F.relu(self.conv7(img))
            tmp[:, i] = self.pool1(img)
            del img
        x = tmp.reshape(x.shape[0], x.shape[1], 4 * 128 * 4)
        del tmp
        self.rnn_out, _ = self.rnn(x)
        x = self.rnn_out.view(x.shape[0], -1)
        self.drop(x)
        x = self.fc(x)
        x = self.max(x)
        return x
 ```
 ## Train
python Train.py
```
 ----------------------------------------------------------------------------------------------------

Begin Training for Patient 1
End Training with        loss: 0.001    Accuracy : 1.000                val-loss: 0.991 val-Accuracy : 0.949

----------------------------------------------------------------------------------------------------
```

# Some Notes:
Tensorflow代码地址：
https://gitee.com/jaegerwang/eeg-learn/tree/master/tf_EEGLearn
Pytorch代码地址：
https://gitee.com/jaegerwang/eeg-learn/tree/master/EEGLearn-Pytorch
