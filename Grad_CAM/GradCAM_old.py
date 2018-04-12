import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from model.nets import nets_factory
import tensorflow.contrib.slim as slim
from model.preprocessing import preprocessing_factory


class GradCAM(object):

    def __init__(self, image_file_name, result_size, result_file_name, num_classes, model_name="resnet_v2_50",
                 layer_conv_name="PrePool", layer_prediction_name="predictions", layer_logits_name="Logits",
                 label_file_name="./data/imagenet/labels.txt", checkpoint_path="./ckpt/resnet_v2_50.ckpt"):

        self.result_size = result_size
        self.num_classes = num_classes
        self.image_file_name = image_file_name
        self.result_file_name = result_file_name
        self.label_file_name = label_file_name
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name
        self.layer_conv_name = layer_conv_name
        self.layer_logits_name = layer_logits_name
        self.layer_prediction_name = layer_prediction_name

        self.inputs, self.end_points = None, None
        self.network_fn = self.get_network_fn(self.model_name, self.num_classes)
        pass

    def init_net(self):
        inputs = tf.placeholder(tf.uint8, [None, None, 3])
        inputs_image = self.preprocess_image(inputs, self.result_size, self.model_name)
        inputs_image = tf.expand_dims(inputs_image, 0)

        _, end_points = self.network_fn(inputs_image)
        return inputs, end_points

    @staticmethod
    def load_labels(label_file):
        labels = {}
        with open(label_file) as label_file:
            for line in label_file:
                idx, label = line.rstrip('\n').split(':')
                labels[int(idx) + 1] = label
            pass
        assert len(labels) > 1
        labels[0] = "000"
        return labels

    @staticmethod
    def load_image(image_input):
        img = cv2.imread(image_input)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            raise Exception("Unable to load img: {}".format(image_input))
        return img

    @staticmethod
    def get_network_fn(model_name, num_classes):
        return nets_factory.get_network_fn(model_name, num_classes=num_classes, is_training=False)

    @staticmethod
    def preprocess_image(image, result_size, preprocessing_name):
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(preprocessing_name, is_training=False)
        image = image_preprocessing_fn(image, result_size, result_size)
        return image

    @staticmethod
    def grad_cam(sess, inputs_x, images, predicted_class, num_classes, conv_layer, logits_layer):
        # 求损失
        one_hot = tf.sparse_to_dense(predicted_class, [num_classes], 1.0)
        loss = tf.reduce_mean(tf.multiply(logits_layer, one_hot))

        # 求梯度
        grads = tf.gradients(loss, conv_layer)[0]
        norm_grads = tf.divide(grads, tf.sqrt(tf.reduce_mean(tf.square(grads))) + tf.constant(1e-5))

        # 运行图
        output, grads_val = sess.run([conv_layer, norm_grads], feed_dict={inputs_x: images})

        # 特征图
        output = output[0]  # [10,10,2048]
        # 梯度
        grads_val = grads_val[0]  # [10,10,2048]

        # 权值：梯度的全局平均
        weights = np.mean(grads_val, axis=(0, 1))  # [2048]

        # 加权和：特征图的加权和
        cam = np.ones(output.shape[0: 2], dtype=np.float32)  # [10,10]
        for i, w in enumerate(weights):
            cam += w * output[:, :, i]

        # ReLU激活
        cam = np.maximum(cam, 0)

        # 归一
        cam = cam / np.max(cam)
        return cam

    def run(self):
        print("Loading image")
        image = self.load_image(self.image_file_name)

        print("Loading labels")
        labels = self.load_labels(self.label_file_name)

        print("Building graph")
        self.inputs, self.end_points = self.init_net()

        print("Loading Model")
        init_fn = slim.assign_from_checkpoint_fn(self.checkpoint_path, slim.get_variables_to_restore())

        with tf.Session() as sess:
            init_fn(sess)

            print("Feed forwarding")
            end_points_result = sess.run(self.end_points, feed_dict={self.inputs: image})
            prediction_values = end_points_result[self.layer_prediction_name][0]
            predicted_class = (np.argsort(prediction_values)[::-1])[0]
            print("Target class: {} {}".format(predicted_class, labels[predicted_class]))

            print("Begin Grad CAM")
            cam = self.grad_cam(sess, self.inputs, image, predicted_class, self.num_classes,
                                conv_layer=self.end_points[self.layer_conv_name],  # 用于可视化的特征图
                                logits_layer=self.end_points[self.layer_logits_name])

            cam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)

            # 上采样
            image = cv2.resize(image, (self.result_size, self.result_size)).astype(float)
            image = image / image.max()
            cam = cv2.resize(cam, (self.result_size, self.result_size))

            alpha = 0.0025
            result = image + alpha * cam
            result = np.asarray(result / np.max(result) * 255, np.uint8)

            im = Image.fromarray(result).convert("RGB")
            im.save(self.result_file_name)
            im.show(result)
        pass

    pass


def main(_):
    grad_cam = GradCAM(image_file_name="./demo/cat.jpg", result_size=300, result_file_name="./demo/output.png",
                       num_classes=1001, model_name="resnet_v2_50",
                       layer_conv_name="PrePool", layer_prediction_name="predictions", layer_logits_name="Logits",
                       label_file_name="./data/imagenet/labels.txt", checkpoint_path="./ckpt/resnet_v2_50.ckpt")
    grad_cam.run()
    pass

if __name__ == '__main__':
    tf.app.run()

