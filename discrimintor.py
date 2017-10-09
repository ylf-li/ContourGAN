def discriminator_simplified_api(inputs, is_train=True, reuse=False):
    df_dim = 64 # Dimension of discrim filters in first conv layer. [64]
    c_dim = FLAGS.c_dim # n_color 3
    batch_size = FLAGS.batch_size # 64
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("discriminator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        net_in = InputLayer(inputs, name='d/in')
        net_h0 = Conv2d(net_in, df_dim, (5, 5), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2),
                padding='SAME', W_init=w_init, name='d/h0/conv2d')

        net_h1 = Conv2d(net_h0, df_dim*2, (5, 5), (2, 2), act=None,
                padding='SAME', W_init=w_init, name='d/h1/conv2d')
        net_h1 = BatchNormLayer(net_h1, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d/h1/batch_norm')

        net_h2 = Conv2d(net_h1, df_dim*4, (5, 5), (2, 2), act=None,
                padding='SAME', W_init=w_init, name='d/h2/conv2d')
        net_h2 = BatchNormLayer(net_h2, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d/h2/batch_norm')

        net_h3 = Conv2d(net_h2, df_dim*8, (5, 5), (2, 2), act=None,
                padding='SAME', W_init=w_init, name='d/h3/conv2d')
        net_h3 = BatchNormLayer(net_h3, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d/h3/batch_norm')

        net_h4 = FlattenLayer(net_h3, name='d/h4/flatten')
        net_h4 = DenseLayer(net_h4, n_units=1, act=tf.identity,
                W_init = w_init, name='d/h4/lin_sigmoid')
        logits = net_h4.outputs
        net_h4.outputs = tf.nn.sigmoid(net_h4.outputs)
return net_h4, logits