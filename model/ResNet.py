from keras.src.layers import Layer, Conv2D, Activation, Add, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D, Dense


class CustomConv2D(Layer):
    def __init__(self, n_filters, kernel_size, n_strides, padding='valid'):
        super(CustomConv2D, self).__init__(name='custom_conv2d')
        self.conv = Conv2D(
            filters=n_filters,
            kernel_size=kernel_size,
            activation='relu',
            strides=n_strides,
            padding=padding,
            kernel_regularizer='l2'
        )
        self.batch_norm = BatchNormalization()
    
    def call(self, x, training=True):
        x = self.conv(x)
        x = self.batch_norm(x)
        return x

class ResidualBlock(Layer):
    def __init__(self, n_channels, n_strides=1):
        super(ResidualBlock, self).__init__(name='res_block')
        self.dotted = (n_strides != 1)
        self.conv_1 = CustomConv2D(n_channels, 3, n_strides, 'same')
        self.conv_2 = CustomConv2D(n_channels, 3, 1, 'same')
        self.activation = Activation('relu')
        if self.dotted:
            self.conv_3 = CustomConv2D(n_channels, 1, n_strides)
    
    def call(self, input, training):
        x = self.conv_1(input, training=training)
        x = self.conv_2(x, training=training)
        if self.dotted:
            x_add = self.conv_3(input, training=training)
            x_add = Add()([x, x_add])
        else:
            x_add = Add()([x, input])
        return self.activation(x_add)

class ResNet(Layer):
    def __init__(self):
        super(ResNet, self).__init__(name='resnet')

        self.conv_1 = CustomConv2D(64, 7, 2, padding='same')
        self.max_pool = MaxPooling2D(3, 2)
    
        self.conv_2_1 = ResidualBlock(64)
        self.conv_2_2 = ResidualBlock(64)
#         self.conv_2_3 = ResidualBlock(64)

        self.conv_3_1 = ResidualBlock(128, 2)
        self.conv_3_2 = ResidualBlock(128)
#         self.conv_3_3 = ResidualBlock(128)
#         self.conv_3_4 = ResidualBlock(128)

        self.conv_4_1 = ResidualBlock(256, 2)
        self.conv_4_2 = ResidualBlock(256)
#         self.conv_4_3 = ResidualBlock(256)
#         self.conv_4_4 = ResidualBlock(256)
#         self.conv_4_5 = ResidualBlock(256)
#         self.conv_4_6 = ResidualBlock(256)

        self.conv_5_1 = ResidualBlock(512, 2)
        self.conv_5_2 = ResidualBlock(512)
#         self.conv_5_3 = ResidualBlock(512)

        self.global_pool = GlobalAveragePooling2D()

        # self.fc_3 = Dense(num_classes, activation='softmax')
    
    def call(self, x, training=True):
        x = self.conv_1(x)
        x = self.max_pool(x)

        x = self.conv_2_1(x, training=training)
        x = self.conv_2_2(x, training=training)
#         x = self.conv_2_3(x, training=training)

        x = self.conv_3_1(x, training=training)
        x = self.conv_3_2(x, training=training)
#         x = self.conv_3_3(x, training=training)
#         x = self.conv_3_4(x, training=training)

        x = self.conv_4_1(x, training=training)
        x = self.conv_4_2(x, training=training)
#         x = self.conv_4_3(x, training=training)
#         x = self.conv_4_4(x, training=training)
#         x = self.conv_4_5(x, training=training)
#         x = self.conv_4_6(x, training=training)

        x = self.conv_5_1(x, training=training)
        x = self.conv_5_2(x, training=training)
#         x = self.conv_5_3(x, training=training)

        x = self.global_pool(x)

#         return self.fc_3(x)
        return x