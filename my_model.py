from keras import layers, models
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint

# 配置GPU，清除之前的缓存
tf.keras.backend.clear_session()
physical_devices = tf.config.list_physical_devices('GPU')
# 逐渐增长，避免电脑GPU内存不足
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# 设置图片目录路径
train_dataset_dir = r'data\spectrogram-dataset1-train\dataset'
test_dataset_dir = r'data\spectrogram-dataset1-test'

# 加载训练集数据
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dataset_dir,
    batch_size=12,
    label_mode='int',
    seed=243,
    image_size=(256, 256),  # 显式设置图片尺寸
)

# 获取输入数据的形状
train_input_shape = train_dataset.element_spec[0].shape[1:]  # (256, 256), 3)
train_num_classes = len(train_dataset.class_names)

# 加载测试数据集
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    test_dataset_dir,
    batch_size=12,
    label_mode='int',
    seed=243,
    image_size=(256, 256),  # 显式设置图片尺寸
)

# 获取测试集输入数据的形状
test_input_shape = test_dataset.element_spec[0].shape[1:]
test_num_classes = len(test_dataset.class_names)


# 加入通道注意力机制
class ChannelAttention(layers.Layer):
    def __init__(self, reduction_ratio=8, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.max_pool = layers.GlobalMaxPooling2D()
        self.dense1 = layers.Dense(input_shape[-1] // self.reduction_ratio, activation='relu')
        self.dense2 = layers.Dense(input_shape[-1], activation='sigmoid')

    def call(self, inputs):
        avg_out = self.dense2(self.dense1(self.avg_pool(inputs)))
        max_out = self.dense2(self.dense1(self.max_pool(inputs)))
        return inputs * tf.expand_dims(tf.expand_dims(avg_out + max_out, axis=1), axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "reduction_ratio": self.reduction_ratio
        })
        return config


# 定义注意力机制
class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.conv = layers.Conv2D(1, kernel_size, activation='sigmoid', padding='same')

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        return inputs * self.conv(concat)

    def get_config(self):
        config = super().get_config()
        config.update({
            "kernel_size": self.kernel_size
        })
        return config


# 定义 CNN 模型
def create_cnn_model(input_shape, num_classes):
    model = models.Sequential()

    # 添加通道注意力机制
    model.add(ChannelAttention())
    # 添加空间注意力机制
    model.add(SpatialAttention())

    # 第一卷积层
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))  # 添加Dropout层，防止过拟合

    # 第二卷积层
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))  # 添加Dropout层，防止过拟合

    # 第三卷积层
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))  # 添加Dropout层，防止过拟合

    # 第四卷积层
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))  # 添加Dropout层，防止过拟合

    # 第五卷积层
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))  # 添加Dropout层，防止过拟合

    # 第六卷积层
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))  # 添加Dropout层，防止过拟合

    # 添加通道注意力机制
    model.add(ChannelAttention())
    # 添加空间注意力机制
    model.add(SpatialAttention())

    # 展平层
    model.add(layers.Flatten())

    # 全连接层
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())  # 全连接层后添加BatchNormalization
    model.add(layers.Dropout(0.3))  # 添加Dropout层，防止过拟合

    # 输出层
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model


# 创建模型
model = create_cnn_model(train_input_shape, train_num_classes)

# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 回调设置：保存最佳模型
model_checkpoint = ModelCheckpoint('my_model_2.h5', save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)

# 训练模型并存储训练过程的历史记录，保存最佳模型
history = model.fit(
    train_dataset,
    epochs=100,
    validation_data=test_dataset,
    callbacks=[model_checkpoint]
)

# 打印报告
model.summary()

# 加载模型
model = keras.models.load_model(r"my_model.h5")

# 绘制准确度曲线
plt.figure(figsize=(12, 4))

# 绘制训练和验证准确度曲线
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# 绘制训练和验证损失曲线
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
