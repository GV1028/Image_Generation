import tensorflow as tf
print(tf.__version__)
import numpy as np
from utils import *
from tensorflow.contrib import learn
import os
from random import shuffle
import sys

class ImageGAN():
    def __init__(self, batch_size = 1, epochs = 10, iterations = 1000, image_shape = [64,64,3], dim_Z = 100, dim_W1 = 1024, dim_W2 = 512, dim_W3 = 256, dim_W4 = 128, dim_W5 = 3):
        self.batch_size = batch_size
        self.epochs = epochs
        self.iterations = iterations
        self.image_shape = image_shape
        self.dim_Z = dim_Z
        self.dim_W1 = dim_W1
        self.dim_W2 = dim_W2
        self.dim_W3 = dim_W3
        self.dim_W4 = dim_W4
        self.dim_W5 = dim_W5

        #Defining generator and discriminator weights
        self.gen1 = tf.Variable(tf.random_normal([dim_Z, dim_W1*4*4], stddev = 0.02), name = 'gen1')
        self.gen2 = tf.Variable(tf.random_normal([5, 5, dim_W2, dim_W1], stddev = 0.02), name = 'gen2')
        #self.gen2b = tf.Variable(tf.constant(0.0, shape = [dim_W1], dtype = tf.float32), trainable = True, name = 'genbias1')
        self.gen3 = tf.Variable(tf.random_normal([5, 5, dim_W3, dim_W2], stddev = 0.02), name = 'gen3')
        self.gen3b = tf.Variable(tf.constant(0.0, shape = [dim_W2], dtype = tf.float32), trainable = True, name = 'genbias2')
        self.gen4 = tf.Variable(tf.random_normal([5, 5, dim_W4, dim_W3], stddev = 0.02), name = 'gen4')
        self.gen4b = tf.Variable(tf.constant(0.0, shape = [dim_W3], dtype = tf.float32), trainable = True, name = 'genbias3')
        self.gen5 = tf.Variable(tf.random_normal([5, 5, dim_W5, dim_W4], stddev = 0.02), name = 'gen5')
        self.gen5b = tf.Variable(tf.constant(0.0, shape = [dim_W4], dtype = tf.float32), trainable = True, name = 'genbias4')

        self.dis1 = tf.Variable(tf.random_normal([5, 5, dim_W5, dim_W4], stddev = 0.02), name = 'dis1')
        self.disb1 = tf.Variable(tf.constant(0.0, shape = [dim_W4], dtype = tf.float32), trainable = True, name = 'disbias1')
        self.dis2 = tf.Variable(tf.random_normal([5, 5, dim_W4, dim_W3], stddev = 0.02), name = 'dis2')
        self.disb2 = tf.Variable(tf.constant(0.0, shape = [dim_W3], dtype = tf.float32), trainable = True, name = 'disbias2')
        self.dis3 = tf.Variable(tf.random_normal([5, 5, dim_W3, dim_W2], stddev = 0.02), name = 'dis3')
        self.disb3 = tf.Variable(tf.constant(0.0, shape = [dim_W2], dtype = tf.float32), trainable = True, name = 'disbias3')
        self.dis4 = tf.Variable(tf.random_normal([5, 5, dim_W2, dim_W1], stddev = 0.02), name = 'dis4')
        self.disb4 = tf.Variable(tf.constant(0.0, shape = [dim_W1], dtype = tf.float32), trainable = True, name = 'disbias4')
        self.dis5 = tf.Variable(tf.random_normal([4*4*dim_W1, 1], stddev = 0.02), name = 'dis5')
        #self.disb5 = tf.Variable(tf.constant(0.0, shape = [4*4*dim_W1], dtype = tf.float32), trainable = True, name = 'disbias5')

        self.d_bn1 = batch_norm(name='dis_bn1')
        self.d_bn2 = batch_norm(name='dis_bn2')
        self.d_bn3 = batch_norm(name='dis_bn3')
        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')

    def generator(self,z):
        with tf.variable_scope("generator") as scope:
            #Project Z
            h1 = lrelu(tf.matmul(z, self.gen1))
            #Reshape
            h1 = tf.reshape(h1, [self.batch_size, 4, 4, self.dim_W1])
            #Upsampling
            genbatch1 = self.g_bn0(h1)
            h2 = lrelu(tf.nn.bias_add(tf.nn.conv2d_transpose(genbatch1, self.gen2, output_shape = [self.batch_size, 8, 8, self.dim_W2], strides = [1, 2, 2, 1]), self.gen3b))
            genbatch2 = self.g_bn1(h2)
            #h2 = tf.nn.dropout(genbatch2, 0.5)
            h3 = lrelu(tf.nn.bias_add(tf.nn.conv2d_transpose(h2, self.gen3, output_shape = [self.batch_size, 16, 16, self.dim_W3], strides = [1, 2, 2, 1]), self.gen4b))
            genbatch3 = self.g_bn2(h3)
            #h3 = tf.nn.dropout(genbatch3, 0.5)
            h4 = lrelu(tf.nn.bias_add(tf.nn.conv2d_transpose(h3, self.gen4, output_shape = [self.batch_size, 32, 32, self.dim_W4], strides = [1, 2, 2, 1]), self.gen5b))
            genbatch4 = self.g_bn3(h4)
            #h4 = tf.nn.dropout(genbatch4, 0.5)
            h5 = tf.nn.conv2d_transpose(h4, self.gen5, output_shape = [self.batch_size, 64, 64, self.dim_W5], strides = [1, 2, 2, 1])
            x = tf.nn.tanh(h5)
            return x

    def discriminator(self, x, reuse = False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            h1 = tf.nn.conv2d(x, self.dis1, strides = [1, 2, 2, 1], padding = 'SAME')
            h1 = lrelu(tf.nn.bias_add(h1, self.disb1))
            h2 = tf.nn.conv2d(h1, self.dis2, strides = [1, 2, 2, 1], padding = 'SAME')
            h2 = self.d_bn1(h2)
            h2 = lrelu(tf.nn.bias_add(h2, self.disb2))
            h3 = tf.nn.conv2d(h2, self.dis3, strides = [1, 2, 2, 1], padding = 'SAME')
            h3 = self.d_bn2(h3)
            h3 = lrelu(tf.nn.bias_add(h3, self.disb3))
            h4 = tf.nn.conv2d(h3, self.dis4, strides = [1, 2, 2, 1], padding = 'SAME')
            h4 = self.d_bn3(h4)
            h4 = lrelu(tf.nn.bias_add(h4, self.disb4))
            h4 = tf.reshape(h4, [self.batch_size, -1])
            h5 = tf.matmul(h4, self.dis5)
            y = tf.nn.sigmoid(h5)

            return y, h5

    def build_model(self):
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.dim_Z])
        self.real_image = tf.placeholder(tf.float32, [self.batch_size] + self.image_shape)
        self.gen_image = self.generator(self.z)
        prob_real, logits_real = self.discriminator(self.real_image)
        prob_fake, logits_fake = self.discriminator(self.gen_image, reuse = True)
        d_real_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits_real, labels = tf.ones_like(prob_real)))
        d_fake_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits_fake, labels = tf.zeros_like(prob_fake)))
        self.g_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits_fake, labels = tf.ones_like(prob_fake)))
        self.d_cost = d_real_cost + d_fake_cost

    def visualize_samples(self, noimages):
        self.zsample = tf.placeholder(tf.float32, [noimages, self.dim_Z])
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()
            #Project Z
            h1 = lrelu(tf.matmul(self.zsample, self.gen1))
            #Reshape
            h1 = tf.reshape(h1, [noimages, 4, 4, self.dim_W1])
            #Upsampling
            genbatch1 = self.g_bn0(h1, train = False)
            h2 = lrelu(tf.nn.bias_add(tf.nn.conv2d_transpose(genbatch1, self.gen2, output_shape = [noimages, 8, 8, self.dim_W2], strides = [1, 2, 2, 1]), self.gen3b))
            genbatch2 = self.g_bn1(h2, train = False)
            #h2 = tf.nn.dropout(genbatch2, 0.5)
            h3 = lrelu(tf.nn.bias_add(tf.nn.conv2d_transpose(h2, self.gen3, output_shape = [noimages, 16, 16, self.dim_W3], strides = [1, 2, 2, 1]), self.gen4b))
            genbatch3 = self.g_bn2(h3, train = False)
            #h3 = tf.nn.dropout(genbatch3, 0.5)
            h4 = lrelu(tf.nn.bias_add(tf.nn.conv2d_transpose(h3, self.gen4, output_shape = [noimages, 32, 32, self.dim_W4], strides = [1, 2, 2, 1]), self.gen5b))
            genbatch4 = self.g_bn3(h4, train = False)
            #h4 = tf.nn.dropout(genbatch4, 0.5)
            h5 = tf.nn.conv2d_transpose(h4, self.gen5, output_shape = [noimages, 64, 64, self.dim_W5], strides = [1, 2, 2, 1])
            x = tf.nn.tanh(h5)
            x = tf.add(x, 1.0)
            x = tf.multiply(x, 0.5*255.0)
            x = tf.to_int32(x)
            return x

    def train(self):
        self.sess = tf.InteractiveSession()
        gen_var = filter(lambda x: x.name.startswith('gen'), tf.trainable_variables())
        dis_var = filter(lambda x: x.name.startswith('dis'), tf.trainable_variables())
        print gen_var
        self.g_opt = tf.train.AdamOptimizer(learning_rate = 0.0002, beta1 = 0.5).minimize(self.g_cost, var_list = gen_var)
        self.d_opt = tf.train.AdamOptimizer(learning_rate = 0.0002, beta1 = 0.5).minimize(self.d_cost, var_list = dis_var)
        noise = np.random.normal(-1, 1, size = [self.batch_size, self.dim_Z]).astype(np.float32)
        switch_count = 100
        visualize_count = 1
        print_count = 500
        noise_sample = np.random.normal(-1, 1, size = [visualize_count, self.dim_Z]).astype(np.float32)
        self.genimage = self.visualize_samples(visualize_count)
        file_path = '/root/code/img_align_celeba/'
        data_files = os.listdir(file_path)
        saver = tf.train.Saver()
        self.ckpt_file = sys.argv[1]
        if self.ckpt_file == "None":
            self.ckpt_file = None
        if self.ckpt_file:
            saver_ = tf.train.import_meta_graph('/root/code/Video_Generation/checkpoints/' + self.ckpt_file + '.meta')
            saver_.restore(self.sess,tf.train.latest_checkpoint('/root/code/Video_Generation/checkpoints/'))
            print "Restored model"
        else:
            tf.global_variables_initializer().run()

        switch_count = 100

        print_count = 25
        noise_sample = np.random.uniform(-1, 1, size = [visualize_count, self.dim_Z]).astype(np.float32)
        file_path = '/root/code/img_align_celeba/'
        data_files = os.listdir(file_path)
        shuffle(data_files)
        for counter in range(len(data_files)/self.batch_size):
            noise = np.random.uniform(-1, 1, size = [self.batch_size, self.dim_Z]).astype(np.float32)
            print("Iteration:", counter)
            batch_files = data_files[counter*self.batch_size:(counter+1)*self.batch_size]
            images = get_real_images(batch_files)
            _, dloss = self.sess.run([self.d_opt, self.d_cost], feed_dict = {self.z : noise, self.real_image : images})
            _, gloss = self.sess.run([self.g_opt, self.g_cost], feed_dict = {self.z : noise})
            _, gloss = self.sess.run([self.g_opt, self.g_cost], feed_dict = {self.z : noise})
            print("Discriminator Loss: ", dloss)
            print("Generator Loss", gloss)

            if np.mod(counter + 1, print_count) == 0:
                gen_images = self.sess.run([self.genimage], feed_dict = {self.zsample : noise_sample})
                gen_images = make_squared_image(gen_images)
                save_gen(gen_images, n_ex = visualize_count, epoch = self.epochs, iter = counter)


        saver.save(self.sess,'/root/code/Video_Generation/checkpoints/VideoGAN_{}_{}_{}.ckpt'.format(self.batch_size,self.epochs,counter))
        print 'Saved {}'.format(counter)

if __name__ == '__main__':
    obj = ImageGAN(batch_size = 64)
    obj.build_model()
    obj.train()
