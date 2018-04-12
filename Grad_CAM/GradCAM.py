import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow.contrib.slim as slim
from GradCAM_old import GradCAM as GradCAM_old


class GradCAM(object):

    def __init__(self, inputs, end_points, image_file_name, result_size, result_file_name, num_classes,
                 checkpoint_path, labels=None):

        self.result_size = result_size
        self.num_classes = num_classes

        self.image_file_name = image_file_name
        self.result_file_name = result_file_name
        self.checkpoint_path = checkpoint_path

        # 标签，可以为None
        self.labels = labels

        # 输入的占位符
        self.inputs = inputs

        # end_points包含三部分：用于可视化的特征图、logits、预测值
        self.layer_conv, self.layer_logits, self.layer_predict = end_points[0], end_points[1], end_points[2]
        pass

    @staticmethod
    def _load_image(image_input):
        img = cv2.imread(image_input)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            raise Exception("Unable to load img: {}".format(image_input))
        return img

    @staticmethod
    def _grad_cam(sess, inputs_x, images, predicted_class, num_classes, conv_layer, logits_layer):
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

    def _core(self, sess, image, run_class, prob):
        # 对指定的类别进行可视化
        print("Target class: {} {} {}".format(run_class, prob,
                                              None if self.labels is None else self.labels[run_class]))

        print("Begin Grad CAM")
        cam = self._grad_cam(sess, self.inputs, image, run_class,
                             self.num_classes, self.layer_conv, self.layer_logits)
        print("End Grad CAM")

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

    def run(self, run_class=None):
        print("Loading image")
        image = self._load_image(self.image_file_name)

        print("Loading Model")
        init_fn = slim.assign_from_checkpoint_fn(self.checkpoint_path, slim.get_variables_to_restore())

        with tf.Session() as sess:
            init_fn(sess)

            prob = None
            if run_class is None:  # 先预测出类别，否则对指定的类别进行可视化
                print("Predicting")
                r_predict = sess.run(self.layer_predict, feed_dict={self.inputs: image})[0]
                predicted_classes = (np.argsort(r_predict)[::-1])
                run_class = predicted_classes[0]
                prob = r_predict[run_class]
                pass

            # 对类别run_class进行可视化
            self._core(sess, image, run_class, prob)
            pass

        pass

    pass


def main(_):
    result_size = 300
    num_classes = 1001
    model_name = "resnet_v2_50"

    # 1.（可选）标签
    labels = GradCAM_old.load_labels(label_file="./data/imagenet/labels.txt")

    # 2.（必选）输入占位符、数据预处理
    inputs = tf.placeholder(tf.uint8, [None, None, 3])
    inputs_image = GradCAM_old.preprocess_image(inputs, result_size, model_name)
    inputs_image = tf.expand_dims(inputs_image, 0)

    # 3.（必选）网络端点：必须包含三部分：用于可视化的特征图、logits、预测值
    _, end_points = GradCAM_old.get_network_fn(model_name, num_classes)(inputs_image)
    need_end_points = [end_points["PrePool"], end_points["Logits"], end_points["predictions"]]

    # 4.准备好前面三部分后，开始可视化
    grad_cam = GradCAM(inputs, need_end_points, image_file_name="./demo/cat.jpg",  num_classes=num_classes,
                       result_size=result_size, result_file_name="./demo/cat_o.png",
                       labels=labels, checkpoint_path="./ckpt/resnet_v2_50.ckpt")
    # 当run_class为某一个类别时，对分类为该类别的像素进行可视化
    grad_cam.run(run_class=None)
    pass

if __name__ == '__main__':
    tf.app.run()
