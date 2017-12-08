import numpy
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import os
import sys
import inspect
import skimage
import tensorflow_test
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def plot_concat_loss():
    loss = numpy.load('tf_test4_loss_record_triple_loss2.npy')

    figure = pyplot.figure()

    for k in range(loss.shape[0]):
        concat = []
        figure = pyplot.figure()
        for i in range(loss.shape[1]):
            concat.append(loss[k, i, :20*i, 0])
        loss_long = numpy.concatenate(tuple(concat))
        pyplot.plot(loss_long)
        figure.savefig('loss_iter'+str(k)+'.png')
        
def plot_loss_all(args):
    dir = args[0]
    is_new_format = True
    if len(args) > 1:
        if args[1] == 'new':
            is_new_format = True
        elif args[1] == 'old':
            is_new_format = False
    plot_each_level = False
    if len(args) > 2 and is_new_format is True:
        if args[2] == 'each':
            plot_each_level = True
        elif args[2] == 'all':
            plot_each_level = False
    log_view = False
    if len(args) > 3:
        if args[3] == 'log':
            log_view = True
        elif args[3] == 'reg':
            log_view = False
    filename = os.path.join(dir, 'tf_test4_loss_record.npy')
    loss = numpy.load(filename)
    loss_all_ind = loss.shape[1] if is_new_format else 1
    if plot_each_level:
        inds = range(loss.shape[3])
    else:
        inds = [loss_all_ind]
    for k in inds:
        figure = pyplot.figure()
        loss_vec = loss[0, :, :, k].reshape(loss.shape[1]*loss.shape[2])
        before_spike = numpy.empty((loss.shape[1], 2))
        after_spike = numpy.empty((loss.shape[1], 2))
        for i in range(loss.shape[1]):
            x_coord = loss.shape[2] * i
            y_coord = loss_vec[x_coord]
            before_spike[i, 0] = x_coord
            before_spike[i, 1] = y_coord
            x_coord += 1
            y_coord = loss_vec[x_coord]
            after_spike[i, 0] = x_coord
            after_spike[i, 1] = y_coord
        if log_view:
            pyplot.plot(numpy.log(loss_vec))
            pyplot.plot(before_spike[:, 0], numpy.log(before_spike[:, 1]), 'bs')
            pyplot.plot(after_spike[:, 0], numpy.log(after_spike[:, 1]), 'g^')
        else:
            pyplot.plot(loss_vec)
            pyplot.plot(before_spike[:, 0], before_spike[:, 1], 'bs')
            pyplot.plot(after_spike[:, 0], after_spike[:, 1], 'g^')
        if k < loss.shape[1]:
            fig_name = os.path.join(dir, 'loss_level'+str(k)+('_log'if log_view else '')+'.png')
        elif k == loss.shape[1]:
            fig_name = os.path.join(dir, 'loss_all_level'+('_log'if log_view else '')+'.png')
            if not log_view:
                pyplot.ylim(0.0, 0.6)
            else:
                pyplot.ylim(-10, 0)
        elif k == loss.shape[1]+1:
            fig_name = os.path.join(dir, 'loss_overlap'+('_log'if log_view else '')+'.png')
        elif k == loss.shape[1]+2:
            fig_name = os.path.join(dir, 'loss_constrain'+('_log'if log_view else '')+'.png')
        elif k == loss.shape[1]+3:
            fig_name = os.path.join(dir, 'loss_train'+('_log'if log_view else '')+'.png')
        figure.savefig(fig_name)
        pyplot.close(figure)
    
def plot_spike_imgs(args):
    dir = args[0]
    filename = os.path.join(dir, 'tf_test4_loss_record.npy')
    loss = numpy.load(filename)
    sess = tf.Session()
    blank_img = tf.zeros(tensorflow_test.size+(3,), dtype=tf.float32)
    tree_var = tf.Variable(tf.random_uniform([len(tensorflow_test.tree)], dtype=tf.float32))
    img, _ = tensorflow_test.render(blank_img, tensorflow_test.tree, tensorflow_test.startpos, from_data=tree_var)
    sess.run(tf.global_variables_initializer())
    for i in range(loss.shape[1]):
        before_spike = numpy.load(os.path.join(dir, '0_'+str(i)+'_0var.npy'))
        after_spike = numpy.load(os.path.join(dir, '0_'+str(i)+'_1var.npy'))
        sess.run(tf.assign(tree_var, before_spike))
        before_img = sess.run(img)
        skimage.io.imsave(os.path.join(dir, 'spike'+str(i)+'before.png'), before_img)
        sess.run(tf.assign(tree_var, after_spike))
        after_img = sess.run(img)
        skimage.io.imsave(os.path.join(dir, 'spike'+str(i)+'cafter.png'), after_img)

def plot_all_gradient(args):
    dir = args[0]
    plot_seperate = False
    if len(args) > 1:
        if args[1] == 'norm':
            plot_seperate = False
        elif args[1] == 'each':
            plot_seperate = True
    filename = os.path.join(dir, 'tf_test4_loss_record.npy')
    loss = numpy.load(filename)
    g_size = numpy.load(os.path.join(dir, '0_0_0var.npy')).shape[0]
    g_vec = numpy.empty([loss.shape[0]*loss.shape[1]*loss.shape[2], g_size if plot_seperate else 1])
    ind = 0
    for i in range(loss.shape[0]):
        for j in range(loss.shape[1]):
            for k in range(loss.shape[2]):
                current_gname = str(i) + '_' + str(j) + '_' + str(k) + 'gradient.npy'
                current_g = numpy.load(os.path.join(dir, current_gname))
                current_g = current_g[:, 0, :].squeeze()
                if plot_seperate:
                    g_vec[ind] = current_g
                else:
                    g_vec[ind, 0] = numpy.linalg.norm(current_g)
                ind += 1
    for v in range(g_vec.shape[1]):
        figure = pyplot.figure()
        pyplot.plot(g_vec[:, v])
        before_x = [i*loss.shape[2]-1 for i in range(1, loss.shape[1])]
        before_y = [g_vec[i, v] for i in before_x]
        after_x = [i+1 for i in before_x]
        after_y = [g_vec[i, v] for i in after_x]
        after_x2 = [i+1 for i in after_x]
        after_y2 = [g_vec[i, v] for i in after_x2]
        pyplot.plot(before_x, before_y, 'bs')
        pyplot.plot(after_x, after_y, 'g^')
        pyplot.plot(after_x2, after_y2, 'ro')
        if plot_seperate:
            fig_name = os.path.join(dir, 'gradient_var'+str(v)+'.png')
        else:
            fig_name = os.path.join(dir, 'gradient_norm.png')
        figure.savefig(fig_name)
        pyplot.close(figure)
        
def plot_everything(args):
    dir = args[0]
    plot_loss_all([dir, 'new', 'each', 'reg'])
    plot_loss_all([dir, 'new', 'each', 'log'])
    plot_spike_imgs([dir])
    plot_all_gradient([dir, 'norm'])
    plot_all_gradient([dir, 'each'])
     
def main():
    func_name = sys.argv[1]
    args = sys.argv[2:]
    eval(func_name)(args)
        
if __name__ == '__main__':
    main()