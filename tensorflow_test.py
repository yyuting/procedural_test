import math
import skimage.io
import tensorflow as tf
import numpy.random
import time
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# incremental color works better, but doesn't fully solve the problem
#increment_color = True

# 10 random restarts doesn't seem to help
random_restarts = 1

rescale = True

loss_type = 'triple'          # one of 'incremental', 'pair', 'triple'
if loss_type == 'incremental':
    increment_color = True
else:
    increment_color = False

#pair_loss = True

#if pair_loss:
#    increment_color = False

use_float32 = True

test_overlap = False

seperate_minimizer = True

rules = {}

rules['1'] = '11'
rules['0'] = '1[0]0'

axiom = '0'
iterations = 3

size = width, height = 1024, 1024

startpos = width - 50, height / 2 
angle = math.pi / 4.0
line_len = 100.0

xv, yv = numpy.meshgrid(numpy.arange(width), numpy.arange(height), indexing='ij')
if use_float32:
    tensor_xv = tf.constant(xv, dtype=tf.float32)
    tensor_yv = tf.constant(yv, dtype=tf.float32)
else:
    tensor_xv = tf.constant(xv, dtype=tf.float64)
    tensor_yv = tf.constant(yv, dtype=tf.float64)

if increment_color:
    b_color = numpy.array([0.3, 0.3, 0.3])
else:
    b_color = numpy.array([1.0, 1.0, 1.0])
l_color = numpy.array([0.0, 1.0, 0.0])

def get_min_dist(points):
    lines = []
    for i in range(len(points)-1):
        x0 = points[i][0]
        y0 = points[i][1]
        x1 = points[i+1][0]
        y1 = points[i+1][1]
        dx = x1 - x0
        dy = y1 - y0
        dist = dx ** 2 + dy ** 2
        line_b = dx * y0 - dy * x0
        lines.append([line_b, dx, dy, dist])
    
    dense_dist = []
    for line_ind in range(len(points)-1):
        k = ((tensor_xv - points[line_ind][0]) * lines[line_ind][1] + \
             (tensor_yv - points[line_ind][1]) * lines[line_ind][2]) / \
            lines[line_ind][3]
        k_clip = tf.clip_by_value(k, 0, 1)
        ux = points[line_ind][0] + k_clip * lines[line_ind][1] - tensor_xv
        uy = points[line_ind][1] + k_clip * lines[line_ind][2] - tensor_yv
        dist_to_line = lines[line_ind][1] * tensor_yv - lines[line_ind][2] * tensor_xv - lines[line_ind][0]
        dist_sign = tf.sign(dist_to_line)
        dense_dist.append(dist_sign * (ux ** 2 + uy ** 2))
    dist_stack = tf.stack(dense_dist, axis=2)
    min_dist = tf.reduce_min(dist_stack, axis=2)
    return min_dist
    
def draw_from_min_dist(input, color, min_dist):
    alpha = tf.clip_by_value(min_dist+1, 0, 1)
    out1 = tf.stack([alpha * color[0], alpha * color[1], alpha * color[2]], axis=2)
    if increment_color:
        return input + out1
    out2 = tf.tile(1 - tf.expand_dims(alpha, axis=2), [1, 1, 3]) * input
    return out1 + out2

def draw_aapolygon(input, points, color):
    min_dist = get_min_dist(points)
    alpha = tf.clip_by_value(min_dist+1, 0, 1)
    out1 = tf.stack([alpha * color[0], alpha * color[1], alpha * color[2]], axis=2)
    out2 = tf.tile(1 - tf.expand_dims(alpha, axis=2), [1, 1, 3]) * input
    if increment_color:
        ans = input + out1
    else:
        ans = out1 + out2
    return ans
    
def loss_from_min_dist_triple(min_dist1, min_dist2, min_dist3):
    return loss_from_min_dist_pair(min_dist1, min_dist2) + \
           loss_from_min_dist_pair(min_dist1, min_dist3) + \
           loss_from_min_dist_pair(min_dist2, min_dist3)
    
def loss_from_min_dist_pair(min_dist1, min_dist2):
    overlap = tf.minimum(min_dist1, min_dist2)
    loss = tf.reduce_sum(tf.clip_by_value(overlap+1, 0, 1))
    return loss
    
def draw_aapolygon_pair(input, points1, points2, color):
    min_dist_list = [get_min_dist(points1), get_min_dist(points2)]
    min_dist = tf.maximum(min_dist_list[0], min_dist_list[1])
    overlap = tf.minimum(min_dist_list[0], min_dist_list[1])
    alpha = tf.clip_by_value(min_dist+1, 0, 1)
    out1 = tf.stack([alpha * color[0], alpha * color[1], alpha * color[2]], axis=2)
    out2 = tf.tile(1 - tf.expand_dims(alpha, axis=2), [1, 1, 3]) * input
    ans = out1 + out2
    loss = tf.reduce_sum(tf.clip_by_value(overlap+1, 0, 1))
    return ans, loss

#sess = tf.Session()
#input1 = tf.zeros(size+(3,), dtype=tf.float64)
#points1 = numpy.array([[10, 10], [1000, 50], [50, 1000], [10, 10]])
#output1 = draw_aapolygon(input1, points1, [1.0, 1.0, 1.0])
#ans1 = sess.run(output1)
#skimage.io.imsave('tf_test.png', ans1)

def apply(var):
    return rules.get(var, var)
    
def recurse(str):
    new_str = ''
    for var in str:
        new_str += apply(var)
    return new_str
    
def build(iters, ax):
    str = ax
    for i in range(iters):
        str = recurse(str)
    return str
    
def polar_to_cart(theta, length, pos):
    x = length * tf.cos(theta)
    y = length * tf.sin(theta)
    return [pos[0] + x, pos[1] + y]
    
def render(img, grammar, pos, linesize=40, dev_angle=math.pi/4, rand=False, from_data=None):
    current_pos = pos
    current_edge = None
    current_angle = math.pi
    stack = []
    loss = 0
    stored_min_dist = [None]*len(grammar)
    stored_left_idx = -numpy.ones(len(grammar)).astype('i')
    for i in range(len(grammar)):
        var = grammar[i]
        if var == '0' or var == '1':
            if rand:
                current_linesize = linesize * (1.0 + numpy.random.randn() / 3.0)
            elif from_data is not None:
                if rescale:
                    current_linesize = from_data[i] * linesize
                else:
                    current_ilnesize = from_data[i]
            else:
                current_linesize = linesize
            if use_float32:
                current_angle = tf.cast(current_angle, tf.float32)
            else:
                current_angle = tf.cast(current_angle, tf.float64)
            new_pos = polar_to_cart(current_angle, line_len, current_pos)
            ang1 = current_angle+math.pi/2.0
            ang2 = current_angle-math.pi/2.0
            if current_edge is None:
                p1 = polar_to_cart(ang1, current_linesize/2.0, current_pos)
                p2 = polar_to_cart(ang2, current_linesize/2.0, current_pos)
                current_edge = [p1, p2]
            else:
                p1 = current_edge[0]
                p2 = current_edge[1]
            p3 = polar_to_cart(ang1, current_linesize/2.0, new_pos)
            p4 = polar_to_cart(ang2, current_linesize/2.0, new_pos)
            new_edge = [p3, p4]
            points = [p1, p2, p4, p3, p1]
            min_dist = get_min_dist(points)
            stored_min_dist[i] = min_dist
            img = draw_from_min_dist(img, b_color, min_dist)
            current_pos = new_pos
            current_edge = new_edge
        elif var == '[' or var == ']':
            if rand:
                current_dev_angle = dev_angle * (1.0 + numpy.random.rand() / 3.0)
                if test_overlap and (i == 4 or i == 19):
                    current_dev_angle /= 2.0
                #current_dev_angle = dev_angle
            elif from_data is not None:
                if rescale:
                    current_dev_angle = from_data[i] * dev_angle
                else:
                    current_dev_angle = from_data[i]
            else:
                current_dev_angle = dev_angle
            if var == '[':
                stack.append((current_pos, current_angle, current_edge, current_linesize, i))
                current_angle -= current_dev_angle
                p1 = polar_to_cart(current_angle+math.pi/2.0, current_linesize/2.0, current_pos)
                p2 = polar_to_cart(current_angle-math.pi/2.0, current_linesize/2.0, current_pos)
                current_edge = [p1, p2]
            else:
                current_pos, current_angle, current_edge, current_linesize, idx = stack.pop()
                current_angle += current_dev_angle
                p1 = polar_to_cart(current_angle+math.pi/2.0, current_linesize/2.0, current_pos)
                p2 = polar_to_cart(current_angle-math.pi/2.0, current_linesize/2.0, current_pos)
                current_edge = [p1, p2]
                if loss_type != 'incremental':
                    left_idx = idx
                    right_idx = i
                    for j in range(idx, len(grammar)):
                        new_var = grammar[j]
                        if new_var == '0' or new_var == '1':
                            left_idx = j
                            break
                    for j in range(i, len(grammar)):
                        new_var = grammar[j]
                        if new_var == '0' or new_var == '1':
                            right_idx = j
                            break
                    stored_left_idx[right_idx] = left_idx
        else:
            raise
    
    for right_idx in range(len(grammar)):
        left_idx = stored_left_idx[right_idx]
        if left_idx >= 0:
            if loss_type == 'pair':
                loss += loss_from_min_dist_pair(stored_min_dist[left_idx], stored_min_dist[right_idx])
            else:
                parent_idx = left_idx
                for j in range(left_idx-1, -1, -1):
                    new_var = grammar[j]
                    if new_var == '1':
                        parent_idx = j
                        break
                loss += loss_from_min_dist_triple(stored_min_dist[left_idx], stored_min_dist[right_idx], stored_min_dist[parent_idx])
            
    if not use_float32:
        loss = tf.cast(loss, tf.float32)
    return img, loss
    
tree = build(iterations, axiom)

def coarse_to_fine_loss(output, ground, nlevels):
    diff = tf.expand_dims(output - ground, axis=0)
    loss = tf.reduce_mean(diff ** 2.0)
    if not use_float32:
        loss = tf.cast(loss, tf.float32)
        node = tf.cast(diff, tf.float32)
    else:
        node = diff
    for n in range(nlevels):
        node = tf.nn.avg_pool(node, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        loss += tf.reduce_mean(node ** 2.0)
    return loss
    
def coarse_to_fine_loss_list(output, ground, nlevels):
    diff = tf.expand_dims(output - ground, axis=0)
    if not use_float32:
        losses = [tf.cast(tf.reduce_mean(diff ** 2.0), tf.float32)]
        node = tf.cast(diff, tf.float32)
    else:
        losses = [tf.reduce_mean(diff ** 2.0)]
        node = diff
    for n in range(nlevels):
        node = tf.nn.avg_pool(node, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        losses.append(tf.reduce_mean(node ** 2.0))
    return losses[::-1]
    
def constrain_width_positive(vars):
    loss = 0
    for i in range(len(tree)):
        if tree[i] == '1' or tree[i] == '0':
            #loss -= tf.sign(vars[i])
            loss += tf.nn.relu(-vars[i])
    if use_float32:
        return loss
    else:
        return tf.cast(loss, tf.float32)

def test1():
    assert loss_type == 'incremental'
    sess = tf.Session()
    if use_float32:
        blank_img = tf.zeros(size+(3,), dtype=tf.float32)
        tree_width = tf.Variable(tf.random_uniform([], dtype=tf.float32))
        tree_angle = tf.Variable(tf.random_uniform([], dtype=tf.float32))
    else:
        blank_img = tf.zeros(size+(3,), dtype=tf.float64)
        tree_width = tf.Variable(tf.random_uniform([], dtype=tf.float64))
        tree_angle = tf.Variable(tf.random_uniform([], dtype=tf.float64))
    ground_img, _ = render(blank_img, tree, startpos)
    ground_arr = sess.run(ground_img)
    skimage.io.imsave('test1_ground.png', ground_arr)
    output, _ = render(blank_img, tree, startpos, 40.0 * tree_width, tree_angle)
    #loss = tf.reduce_mean((output - ground_img) ** 2)
    loss = coarse_to_fine_loss(output, ground_img, 9)
    minimizer = tf.train.AdamOptimizer(learning_rate=0.1)
    #minimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    step = minimizer.minimize(loss)
    sess.run(tf.global_variables_initializer())
    init_arr = sess.run(output)
    skimage.io.imsave('test1_start.png', init_arr)
    for i in range(50):
        _, v_width, v_angle, v_loss = sess.run([step, tree_width, tree_angle, loss])
        print(v_loss, v_angle, v_width)
    output_arr = sess.run(output)
    skimage.io.imsave('test1_output.png', output_arr)
        
#test1()

def test2():
    assert loss_type == 'incremental'
    sess = tf.Session()
    if use_float32:
        blank_img = tf.zeros(size+(3,), dtype=tf.float32)
        tree_var = tf.Variable(tf.random_uniform([len(tree)], dtype=tf.float32))
    else:
        blank_img = tf.zeros(size+(3,), dtype=tf.float64)
        tree_var = tf.Variable(tf.random_uniform([len(tree)], dtype=tf.float64))
    ground_img, _ = render(blank_img, tree, startpos, rand=True)
    ground_arr = numpy.clip(sess.run(ground_img), 0.0, 1.0)
    #skimage.io.imsave('tf_test2_ground.png', ground_arr)
    output, _ = render(blank_img, tree, startpos, from_data=tree_var)
    loss = coarse_to_fine_loss(output, ground_arr, 9)
    #loss -= 0.0001 * tf.cast(tf.reduce_sum(tree_var), tf.float32)
    constrain = constrain_width_positive(tree_var)
    minimizer = tf.train.AdamOptimizer(learning_rate=0.1)
    #minimizer = tf.train.GradientDescentOptimizer(learning_rate=10.0)
    step = minimizer.minimize(loss + constrain)
    sess.run(tf.global_variables_initializer())
    init_arr = sess.run(output)
    #skimage.io.imsave('tf_test2_start.png', numpy.clip(init_arr, 0.0, 1.0))
    best_loss = 1e8
    best_output = None
    for _ in range(random_restarts):
        for i in range(2000):
            _, v_var, v_loss = sess.run([step, tree_var, loss])
            print(v_loss, v_var)
            if v_loss < best_loss:
                best_loss = v_loss
                best_output = sess.run(output)
                numpy.save('tf_test2.npy', v_var)
        sess.run(tf.global_variables_initializer())
    #output_arr = sess.run(output)
    skimage.io.imsave('tf_test2_output.png', numpy.clip(best_output, 0.0, 1.0))

#test2()
#print(tree)

def test3():
    assert loss_type == 'incremental'
    overlap_scale = 1e-5
    sess = tf.Session()
    if use_float32:
        blank_img = tf.zeros(size+(3,), dtype=tf.float32)
        tree_var = tf.Variable(tf.random_uniform([len(tree)], dtype=tf.float32))
    else:
        blank_img = tf.zeros(size+(3,), dtype=tf.float64)
        tree_var = tf.Variable(tf.random_uniform([len(tree)], dtype=tf.float64))
    ground_img, overlap_loss = render(blank_img, tree, startpos, rand=True)
    ground_arr = numpy.clip(sess.run(ground_img), 0.0, 1.0)
    skimage.io.imsave('tf_test3_ground.png', ground_arr)
    output, overlap_loss = render(blank_img, tree, startpos, from_data=tree_var)
    loss = coarse_to_fine_loss(output, ground_arr, 9)
    constrain = constrain_width_positive(tree_var)
    minimizer = tf.train.AdamOptimizer(learning_rate=0.1)
    step = minimizer.minimize(loss + constrain + overlap_loss * overlap_scale)
    #optimizer = tf.contrib.opt.ScipyOptimizerInterface(1e4*(loss+overlap_loss+constrain), method='BFGS')
    #with tf.Session() as session:
    #    session.run(tf.global_variables_initializer())
    #    optimizer.minimize(session)
    #    v_var, v_loss, v_oloss, v_out = session.run([tree_var, loss, overlap_loss, output])
    sess.run(tf.global_variables_initializer())
    init_arr = sess.run(output)
    skimage.io.imsave('tf_test3_start.png', numpy.clip(init_arr, 0.0, 1.0))
    best_loss = 1e8
    best_output = None
    for _ in range(random_restarts):
        for i in range(2000):
            _, v_var, v_loss, v_oloss = sess.run([step, tree_var, loss, overlap_loss])
            print(v_loss, v_oloss, v_var)
            if v_loss < best_loss:
                best_loss = v_loss
                best_output = sess.run(output)
                numpy.save('tf_test3.npy', v_var)
        sess.run(tf.global_variables_initializer())
    skimage.io.imsave('tf_test3_output.png', numpy.clip(best_output, 0.0, 1.0))
    
#test3()

def test4():
    nlevels = 9
    overlap_scale = 1e-6
    sess = tf.Session()
    if use_float32:
        blank_img = tf.zeros(size+(3,), dtype=tf.float32)
        tree_var = tf.Variable(tf.random_uniform([len(tree)], dtype=tf.float32))
    else:
        blank_img = tf.zeros(size+(3,), dtype=tf.float64)
        tree_var = tf.Variable(tf.random_uniform([len(tree)], dtype=tf.float64))
    ground_img, overlap_loss = render(blank_img, tree, startpos, rand=True)
    ground_arr = numpy.clip(sess.run(ground_img), 0.0, 1.0)
    skimage.io.imsave('tf_test4_ground_large_angle.png', ground_arr)
    output, overlap_loss = render(blank_img, tree, startpos, from_data=tree_var)
    losses = coarse_to_fine_loss_list(output, ground_arr, nlevels)
    constrain = constrain_width_positive(tree_var)
    if loss_type != 'incremental':
        constrain += overlap_loss * overlap_scale
    minimizer = tf.train.AdamOptimizer(learning_rate=0.1)
    steps = []
    loss_all = 0
    for loss in losses:
        if seperate_minimizer:
            steps.append(tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss + constrain))
        else:
            steps.append(minimizer.minimize(loss + constrain))
        loss_all += loss
    sess.run(tf.global_variables_initializer())
    init_arr = sess.run(output)
    skimage.io.imsave('tf_test4_start_large_angle.png', numpy.clip(init_arr, 0.0, 1.0))
    best_loss = 1e8
    best_output = None
    best_var = None
    iter_i = 5
    iter_n = len(steps)
    iter_k = 20
    loss_record = numpy.empty((iter_i, iter_n, iter_k*iter_n, 2))
    for _ in range(random_restarts):
        for i in range(iter_i):
            print("i,", i)
            for n in range(iter_n):
                print("n,", n)
                for k in range(iter_k*(n+1)):
                    step = steps[n]
                    loss = losses[n]
                    _, v_var, v_loss, v_loss_all = sess.run([step, tree_var, loss, loss_all])
                    print(v_loss, v_loss_all)
                    loss_record[i, n, k, 0] = v_loss
                    loss_record[i, n, k, 1] = v_loss_all
                    if v_loss_all < best_loss:
                        best_loss = v_loss_all
                        best_var = v_var
                        numpy.save('tf_test4.npy', v_var)
        sess.run(tf.global_variables_initializer())
    sess.run(tf.assign(tree_var, best_var))
    if seperate_minimizer:
        numpy.save('tf_test4_loss_record_large_angle.npy', loss_record)
    else:
        numpy.save('tf_test4_loss_record.npy', loss_record)
    best_output = sess.run(output)
    skimage.io.imsave('tf_test4_output_large_angle.png', numpy.clip(best_output, 0.0, 1.0))

test4()
    
#blank_img = tf.zeros(size+(3,), dtype=tf.float64)
#var = numpy.load('tf_test2.npy')
#output = render(blank_img, tree, startpos, from_data=var)
#sess = tf.Session()
#sess.run(tf.global_variables_initializer())
#skimage.io.imsave('test.png', sess.run(output))