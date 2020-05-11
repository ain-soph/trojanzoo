import onnx
from onnx2keras import onnx_to_keras
import tensorflow as tf
onnx_model = onnx.load('model.onnx')
k_model = onnx_to_keras(onnx_model, ['input'])
tf.keras.models.save_model(k_model, "./model.h5")