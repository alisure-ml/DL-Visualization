import multiprocessing
import numpy as np
import os
import alisuretool as alitool
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def get_data(data_filename, label_filename, select_class):
    data = np.load(data_filename)
    test_label = np.load(label_filename)

    classes = sorted(list(set(test_label)))
    # print(classes)
    labels = np.asarray([classes.index(v) for v in test_label])

    data_new = []
    label_new = []
    for index, select_class_one in enumerate(select_class):
        data_new.extend(data[labels == select_class_one])
        label_new.extend(np.zeros_like(labels[labels == select_class_one], dtype=np.int) + index)
        pass

    data_new = np.asarray(data_new)
    label_new = np.asarray(label_new)
    n_samples, n_features = data_new.shape

    return data_new, label_new, n_samples, n_features


def plot_embedding(data, label, title, s=None):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    color = "bgrcmyk"
    fig = plt.figure()
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], s if s is not None else str(label[i]),
                 color=color[label[i]], fontdict={'weight': 'bold', 'size': 6})
        pass

    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig


def main(data_filename, label_filename, select_class, result_png, title, s):
    data, label, n_samples, n_features = get_data(data_filename, label_filename, select_class)
    print('Computing t-SNE embedding {} {} {}'.format(data_filename, label_filename, result_png))
    tsne = TSNE(n_components=2)
    result = tsne.fit_transform(data)
    print("begin to embedding {}".format(result_png))
    fig = plot_embedding(result, label, title, s)
    print("begin to save {}".format(result_png))
    plt.savefig(result_png)
    # plt.show(fig)
    pass


def class_name(label_filename):
    for label_filename_one in label_filename:
        test_label = np.load(label_filename_one)
        classes = sorted(list(set(test_label)))
        print(classes)
    pass


"""
['airplane', 'alarm_clock', 'ant', 'ape', 'apple', 'armor', 'axe', 'banana', 'bear', 'bee', 'beetle', 'bell', 'bench',
 'bicycle', 'blimp', 'bread', 'butterfly', 'camel', 'candle', 'cannon', 'car_(sedan)', 'castle', 'cat', 'chair',
 'chicken', 'church', 'couch', 'crab', 'crocodilian', 'cup', 'deer', 'dog', 'duck', 'elephant', 'eyeglasses', 'fan',
 'fish', 'flower', 'frog', 'geyser', 'guitar', 'hamburger', 'hammer', 'harp', 'hat', 'hedgehog', 'hermit_crab', 'horse',
 'hot-air_balloon', 'hotdog', 'hourglass', 'jack-o-lantern', 'jellyfish', 'kangaroo', 'knife', 'lion', 'lizard',
 'lobster', 'motorcycle', 'mushroom', 'owl', 'parrot', 'penguin', 'piano', 'pickup_truck', 'pig', 'pineapple', 'pistol',
 'pizza', 'pretzel', 'rabbit', 'racket', 'ray', 'rifle', 'rocket', 'sailboat', 'saxophone', 'scorpion', 'sea_turtle',
 'seal', 'shark', 'sheep', 'shoe', 'snail', 'snake', 'spider', 'spoon', 'squirrel', 'starfish', 'strawberry', 'swan',
 'table', 'tank', 'teapot', 'teddy_bear', 'tiger', 'trumpet', 'turtle', 'umbrella', 'violin', 'volcano', 'wading_bird',
 'wine_bottle', 'zebra']
 
 0 蝙蝠 小屋 奶牛 海豚 门
 5 长颈鹿 直升机 老鼠 梨 浣熊
 10 犀牛 锯 剪刀 海豚 摩天大楼
 15 鸣鸟 剑 树 轮椅 风车
 20 窗户
['bat', 'cabin', 'cow', 'dolphin', 'door', 'giraffe', 'helicopter', 'mouse', 'pear', 'raccoon', 'rhinoceros', 'saw', 
'scissors', 'seagull', 'skyscraper', 'songbird', 'sword', 'tree', 'wheelchair', 'windmill', 'window']
"""


if __name__ == '__main__':

    data_filename = ['./test/train_image.npy', './test/train_sketch.npy',
                     './test/test_sketch.npy', './test/test_image.npy',
                     './test/ext_image.npy', './test/predict_image.npy']
    label_filename = ['./test/train_image_class.npy', './test/train_image_class.npy',
                      './test/test_image_class.npy', './test/test_image_class.npy',
                      './test/ext_image_class.npy', './test/test_image_class.npy']
    titles = ["t-SNE on Sketchy of Seen Classes", "t-SNE on Sketchy of Seen Classes",
              "t-SNE on Sketchy of Unseen Classes", "t-SNE on Sketchy of Unseen Classes",
              "t-SNE on Sketchy of Seen Classes", "t-SNE on Sketchy of Unseen Classes"]
    # class_name(label_filename)
    select_class = [[5, 12, 16, 19]]
    # select_class = [[4, 14, 20]]
    # select_class = [[5, 15, 10, 14, 19]]

    cpu_count = multiprocessing.cpu_count() - 2
    task_count = len(data_filename) * len(select_class)

    pool = multiprocessing.Pool(processes=cpu_count if cpu_count < task_count else task_count)
    for index, data_filename_one in enumerate(data_filename):
        if index < 2:
            continue
        for class_index, select_class_one in enumerate(select_class):
            result_filename = "./alisure/{}/1_{}_{}_{}.png".format(
                "_".join([str(c) for c in select_class_one]),
                os.path.splitext(os.path.basename(data_filename[index]))[0],
                os.path.splitext(os.path.basename(label_filename[index]))[0],
                "".join([str(c) for c in select_class_one]))
            alitool.Tools.new_dir(result_filename)

            pool.apply_async(main, args=(data_filename[index], label_filename[index],
                                         select_class_one, result_filename, titles[index], "*"))
            # main(data_filename[index], label_filename[index], select_class_one, result_filename, titles[index])
        pass

    pool.close()
    pool.join()

    pass
