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
## 生成EEG图像
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
