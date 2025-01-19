import keras
import tensorflow as tf

# 配置GPU，清除之前的缓存
tf.keras.backend.clear_session()
physical_devices = tf.config.list_physical_devices('GPU')

# 逐渐增长，避免电脑GPU内存不足
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

test_dataset_dir = r'data\spectrogram-dataset1-test'
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

# 加载模型
model = keras.models.load_model(r"my_model.h5")

# 在测试集上评估模型的准确度
loss, accuracy = model.evaluate(test_dataset)

print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")
