from util import model
import argparse
import random
import tensorflow as tf
from util import loader as ld
from util import repoter as rp

directory_path = r'C:/Users/k-higuchi\Desktop\LAB2019\2019_04_10\Middle_Data\quarter_resolution\MiddEval3-data-Q\MiddEval3\trainingQ'

#loader.pyを使ってる
def load_dataset(train_rate):
    loader = ld.Loader(directory_path=directory_path)
    return loader.load_train_test(train_rate=train_rate, shuffle=False)


def train(parser):
    train, test = load_dataset(train_rate=parser.trainrate)
    valid = train.perm(0, 10)
    test = test.perm(0, 20)

    # 結果保存用のインスタンスを作成します
    # Create Reporter Object
    reporter = rp.Reporter(parser=parser)
    accuracy_fig = reporter.create_figure(
        "Accuracy", ("epoch", "accuracy"), ["train", "test"])
    loss_fig = reporter.create_figure(
        "Loss", ("epoch", "loss"), ["train", "test"])

    
    #GPU設定
    gpu = parser.gpu
    gpu_config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7), device_count={'GPU': 0},
                                log_device_placement=False, allow_soft_placement=True)
    # モデル設定
    #300とかにすると割り切れなくてダメみたい
    model_unet = model.UNet(size=(128,128),l2_reg=parser.l2reg).model

    # loss関数acc設定
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=model_unet.teacher,
                                                                           logits=model_unet.outputs))
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

    correct_prediction = tf.equal(
        tf.argmax(model_unet.outputs, 3), tf.argmax(model_unet.teacher, 3))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



    sess = tf.InteractiveSession(config=gpu_config) if gpu else tf.InteractiveSession()
    tf.global_variables_initializer().run()
    
    #parserから設定、epoch,batch_size,augmentation
    epochs = parser.epoch
    batch_size = parser.batchsize
    is_augment = parser.augmentation
    
    #trainを
    train_dict = {model_unet.inputs: valid.images_left, model_unet.teacher: valid.images_right,
                  model_unet.is_training: False}
    test_dict = {model_unet.inputs: test.images_left, model_unet.teacher: test.images_right,
                 model_unet.is_training: False}
    
    for epoch in range(epochs):
        for batch in train(batch_size=batch_size, augment=is_augment):
            # バッチデータの展開
            inputs = batch.images_left
            teacher = batch.images_right
            # Training
            sess.run(train_step, feed_dict={model_unet.inputs: inputs, model_unet.teacher: teacher,
                                            model_unet.is_training: True})
        if epoch % 1 == 0:
            loss_train = sess.run(cross_entropy, feed_dict=train_dict)
            loss_test = sess.run(cross_entropy, feed_dict=test_dict)
            accuracy_train = sess.run(accuracy, feed_dict=train_dict)
            accuracy_test = sess.run(accuracy, feed_dict=test_dict)
            print("Epoch:", epoch)
            print("[Train] Loss:", loss_train, " Accuracy:", accuracy_train)
            print("[Test]  Loss:", loss_test, "Accuracy:", accuracy_test)
            accuracy_fig.add([accuracy_train, accuracy_test], is_update=True)
            loss_fig.add([loss_train, loss_test], is_update=True)
            if epoch % 3 == 0:
                idx_train = random.randrange(10)
                idx_test = random.randrange(3)
                outputs_train = sess.run(model_unet.outputs,
                                         feed_dict={model_unet.inputs: [train.images_left[idx_train]],
                                                    model_unet.is_training: False})
                outputs_test = sess.run(model_unet.outputs,
                                        feed_dict={model_unet.inputs: [test.images_left[idx_test]],
                                                   model_unet.is_training: False})
                train_set = [train.images_left[idx_train],
                             outputs_train[0], train.images_right[idx_train]]
                test_set = [test.images_left[idx_test],
                            outputs_test[0], test.images_right[idx_test]]
                reporter.save_image_from_ndarray(train_set, test_set, train.palette, epoch)

    loss_test = sess.run(cross_entropy, feed_dict=test_dict)
    accuracy_test = sess.run(accuracy, feed_dict=test_dict)
    print("Result")
    print("[Test]  Loss:", loss_test, "Accuracy:", accuracy_test)

    sess.close()


def get_parser():
    parser = argparse.ArgumentParser(
        prog='Image segmentation using U-Net',
        usage='python main.py',
        description='This module demonstrates image segmentation using U-Net.',
        add_help=True
    )

    parser.add_argument('-g', '--gpu', action='store_true', help='Using GPUs')
    parser.add_argument('-e', '--epoch', type=int,
                        default=8, help='Number of epochs')
    parser.add_argument('-b', '--batchsize', type=int,
                        default=4, help='Batch size')
    parser.add_argument('-t', '--trainrate', type=float,
                        default=0.85, help='Training rate')
    parser.add_argument('-a', '--augmentation',
                        action='store_true', help='Number of epochs')
    parser.add_argument('-r', '--l2reg', type=float,
                        default=0.0001, help='L2.')

    return parser


if __name__ == '__main__':
    parser = get_parser().parse_args()
    train(parser)
