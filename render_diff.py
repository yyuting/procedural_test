import math
import numpy
import skimage.io
import scipy.optimize
import numpy.random
import time
import pyximport; pyximport.install()
import cython_render_diff
import sys

rescale = True
rescale_half_pi = False
random_restarts = 1

draw_aapolygon = cython_render_diff.draw_aapolygon
b_color = numpy.array([0.3, 0.3, 0.3])
draw_aapolygon = cython_render_diff.draw_aapolygon_inc
#b_color = numpy.array([1.0, 1.0, 1.0])
l_color = numpy.array([0.0, 1.0, 0.0])

def draw_aapolygon_dummy(img, points, color):
    """
    draws anti-aliased polygon
    img: numpy image array, that contains pixel values
    points: nx2 numpy array that contains polygon 
            points must be in a counter-clockwise order
    color: color set to the polygon
    """
    # find bounding box for effected area
    bbox_xmax = math.ceil(numpy.max(points[:, 0]))
    bbox_xmin = math.floor(numpy.min(points[:, 0]))
    bbox_ymax = math.ceil(numpy.max(points[:, 1]))
    bbox_ymin = math.floor(numpy.min(points[:, 1]))
    
    # calculate line as y = kx + b from x0, y0 and x1, y1
    B = 0
    DX = 1
    DY = 2
    DIST = 3
    lines = numpy.empty((points.shape[0]-1, 5))
    for i in range(points.shape[0]-1):
        x0 = points[i, 0]
        y0 = points[i, 1]
        x1 = points[i+1, 0]
        y1 = points[i+1, 1]
        dx = x1 - x0
        dy = y1 - y0
        dist = dx ** 2 + dy ** 2
        line_b = dx * y0 - dy * x0
        lines[i, B] = line_b
        lines[i, DX] = dx
        lines[i, DY] = dy
        lines[i, DIST] = dist
    
    def intersect_cell(x, y, line_ind, int_x):
        if int_x:
            int_p = (x, lines[line_ind, K] * x + lines[line_ind, B])
        else:
            int_p = ((y - lines[line_ind, B]) / lines[line_ind, K], y)
        k = ((int_p[0] - points[line_ind, 0]) * lines[line_ind, DX] + \
             (int_p[1] - points[line_ind, 1]) * lines[line_ind, DY]) / \
             lines[line_ind, DIST]
        if k >= 0 and k <= 1:
            if int_x:
                return int_p[1] - y
            else:
                return int_p[0] - x
        
    for xx in range(bbox_xmin-1, bbox_xmax+2):
        for yy in range(bbox_ymin-1, bbox_ymax+2):
            if xx == 30 and yy == 300:
                xx = 30
            x = xx + 0.5
            y = yy + 0.5
            all_dists = numpy.empty(points.shape[0]-1)
            for line_ind in range(points.shape[0]-1):
                k = ((x - points[line_ind, 0]) * lines[line_ind, DX] + \
                     (y - points[line_ind, 1]) * lines[line_ind, DY]) / \
                     lines[line_ind, DIST]
                if k < 0:
                    k = 0
                if k >= 1:
                    k = 1
                ux = points[line_ind, 0] + k * lines[line_ind, DX] - x
                uy = points[line_ind, 1] + k * lines[line_ind, DY] - y
                dist_sign = numpy.sign(lines[line_ind, DX] * y - lines[line_ind, DY] * x - lines[line_ind, B])
                if dist_sign == 0:
                    dist_sign = -1
                all_dists[line_ind] = dist_sign * (ux ** 2 + uy ** 2) ** 0.5
            min_dist = numpy.min(all_dists)
            if min_dist + 1 >= 1:
                img[xx, yy] = color
            elif min_dist + 1 > 0:
                alpha = min_dist + 1
                background_color = img[xx, yy]
                img[xx, yy] = alpha * color + (1 - alpha) * background_color
    return

#img1 = numpy.zeros((512, 512, 3), dtype=numpy.float64)
#points1 = numpy.array([[10, 10], [50, 300], [30, 500], [10, 10]])
#img1 = draw_aapolygon(img1, points1, numpy.array([1.0, 1.0, 1.0]))
#img2 = numpy.zeros(img1.shape, dtype=numpy.float64)
#points2 = points1 + 1e-10
#img2 = draw_aapolygon(img2, points2, numpy.array([1.0, 1.0, 1.0]))
#diff = img1 - img2
#print(numpy.sum(diff))
#skimage.io.imsave('test.png', img)

rules = {}

rules['1'] = '11'
rules['0'] = '1[0]0'

axiom = '0'
iterations = 3

size = width, height = 1024, 1024

startpos = width - 50, height / 2 
angle = math.pi / 4.0
line_len = 100.0

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
    x = length * math.cos(theta)
    y = length * math.sin(theta)
    return [pos[0] + x, pos[1] + y]
 
def render_prealloc(img, grammar, pos, linesize=40, dev_angle=math.pi/4, rand=False, from_data=None):
    img[:] = 0.0
    current_pos = pos
    current_edge = None
    current_angle = math.pi
    stack = []
    for i in range(len(grammar)):
        var = grammar[i]
        if var == '0' or var == '1':
            if rand:
                #current_line_len = line_len * (1.0 + numpy.random.randn() / 5.0)
                current_linesize = linesize * (1.0 + numpy.random.randn() / 5.0)
            elif from_data is not None:
                if rescale:
                    current_linesize = from_data[i] * linesize
                else:
                    current_linesize = from_data[i]
            else:
                #current_line_len = line_len
                current_linesize= linesize
            new_pos = polar_to_cart(current_angle, line_len, current_pos)
            if current_edge is None:
                p1 = polar_to_cart(current_angle+math.pi/2.0, current_linesize/2.0, current_pos)
                p2 = polar_to_cart(current_angle-math.pi/2.0, current_linesize/2.0, current_pos)
                current_edge = [p1, p2]
            else:
                p1 = current_edge[0]
                p2 = current_edge[1]
            p3 = polar_to_cart(current_angle+math.pi/2.0, current_linesize/2.0, new_pos)
            p4 = polar_to_cart(current_angle-math.pi/2.0, current_linesize/2.0, new_pos)
            new_edge = [p3, p4]
            points = numpy.array([p1, p2, p4, p3, p1])
            draw_aapolygon(img, points, b_color)
            current_pos = new_pos
            current_edge = new_edge
        elif var == '[' or var == ']':
            if rand:
                current_dev_angle = dev_angle * (1.0 + numpy.random.rand() / 5.0)
            elif from_data is not None:
                if rescale:
                    current_dev_angle = from_data[i] * dev_angle
                else:
                    current_dev_angle = from_data[i]
            else:
                current_dev_angle = dev_angle
            if var == '[':
                stack.append((current_pos, current_angle, current_edge, current_linesize))
                current_angle -= current_dev_angle
                p1 = polar_to_cart(current_angle+math.pi/2.0, current_linesize/2.0, current_pos)
                p2 = polar_to_cart(current_angle-math.pi/2.0, current_linesize/2.0, current_pos)
                current_edge = [p1, p2]
            else:
                current_pos, current_angle, current_edge, current_linesize = stack.pop()
                current_angle += current_dev_angle
                p1 = polar_to_cart(current_angle+math.pi/2.0, current_linesize/2.0, current_pos)
                p2 = polar_to_cart(current_angle-math.pi/2.0, current_linesize/2.0, current_pos)
                current_edge = [p1, p2]
        else:
            raise
    return

def render_overlap_loss(img, grammar, pos, linesize=40, dev_angle=math.pi/4, rand=False, from_data=None, var_len=False):
    img[:] = 0.0
    current_pos = pos
    current_edge = None
    current_angle = math.pi
    stack = []
    stored_points = []
    paired_idx = numpy.zeros(len(grammar), dtype=int)
    data_idx = 0
    tree_len = len(tree)
    for i in range(len(grammar)):
        var = grammar[i]
        if var == '0' or var == '1':
            if rand:
                current_linesize = linesize * (1.0 + numpy.random.randn() / 5.0)
                if var_len:
                    current_linelen = line_len * (1.0 + numpy.random.randn() / 5.0)
                else:
                    current_linelen = line_len
            elif from_data is not None:
                if rescale:
                    current_linesize = from_data[i] * linesize
                    if var_len:
                        current_linelen = from_data[data_idx+tree_len] * line_len
                    else:
                        current_linelen = line_len
                else:
                    current_linesize = from_data[i]
                    if var_len:
                        current_linelen = from_data[data_idx+tree_len]
                    else:
                        current_linelen = line_len
                if var_len:
                    data_idx += 1
            else:
                current_linesize= linesize
                current_linelen = line_len
            new_pos = polar_to_cart(current_angle, current_linelen, current_pos)
            if current_edge is None:
                p1 = polar_to_cart(current_angle+math.pi/2.0, current_linesize/2.0, current_pos)
                p2 = polar_to_cart(current_angle-math.pi/2.0, current_linesize/2.0, current_pos)
                current_edge = [p1, p2]
            else:
                p1 = current_edge[0]
                p2 = current_edge[1]
            p3 = polar_to_cart(current_angle+math.pi/2.0, current_linesize/2.0, new_pos)
            p4 = polar_to_cart(current_angle-math.pi/2.0, current_linesize/2.0, new_pos)
            new_edge = [p3, p4]
            points = numpy.array([p1, p2, p4, p3, p1])
            stored_points.append(points)
            current_pos = new_pos
            current_edge = new_edge
        elif var == '[' or var == ']':
            if rand:
                current_dev_angle = dev_angle * (1.0 + numpy.random.rand() / 5.0)
            elif from_data is not None:
                if rescale:
                    if rescale_half_pi:
                        current_dev_angle = from_data[i] * dev_angle
                    else:
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
                paired_idx[left_idx] = right_idx
                paired_idx[right_idx] = -1
            stored_points.append([])
        else:
            raise
    
    loss = 0
    for i in range(len(grammar)):
        var = grammar[i]
        if var == '0' or var == '1':
            #draw_aapolygon(img, stored_points[i], b_color)
            #continue
            if paired_idx[i] < 0:
                continue
            elif paired_idx[i] > 0:
                loss += cython_render_diff.draw_aapolygon_overlap_loss(img, stored_points[i], stored_points[paired_idx[i]], b_color)
            else:
                draw_aapolygon(img, stored_points[i], b_color)
    
    return loss
    
def render(grammar, pos, linesize=40, dev_angle=math.pi/4, rand=False, from_data=None):
    current_pos = pos
    current_edge = None
    current_angle = math.pi
    stack = []
    img = numpy.zeros(size+(3,), dtype=numpy.float64)
    for i in range(len(grammar)):
        var = grammar[i]
        if var == '0' or var == '1':
            if rand:
                #current_line_len = line_len * (1.0 + numpy.random.randn() / 5.0)
                current_linesize = linesize * (1.0 + numpy.random.randn() / 5.0)
            elif from_data is not None:
                if rescale:
                    current_linesize = from_data[i] * linesize
                else:
                    current_linesize = from_data[i]
            else:
                #current_line_len = line_len
                current_linesize= linesize
            new_pos = polar_to_cart(current_angle, line_len, current_pos)
            if current_edge is None:
                p1 = polar_to_cart(current_angle+math.pi/2.0, current_linesize/2.0, current_pos)
                p2 = polar_to_cart(current_angle-math.pi/2.0, current_linesize/2.0, current_pos)
                current_edge = [p1, p2]
            else:
                p1 = current_edge[0]
                p2 = current_edge[1]
            p3 = polar_to_cart(current_angle+math.pi/2.0, current_linesize/2.0, new_pos)
            p4 = polar_to_cart(current_angle-math.pi/2.0, current_linesize/2.0, new_pos)
            new_edge = [p3, p4]
            points = numpy.array([p1, p2, p4, p3, p1])
            draw_aapolygon(img, points, b_color)
            current_pos = new_pos
            current_edge = new_edge
        elif var == '[' or var == ']':
            if rand:
                current_dev_angle = dev_angle * (1.0 + numpy.random.rand() / 5.0)
            elif from_data is not None:
                if rescale:
                    current_dev_angle = from_data[i] * dev_angle
                else:
                    current_dev_angle = from_data[i]
            else:
                current_dev_angle = dev_angle
            if var == '[':
                stack.append((current_pos, current_angle, current_edge, current_linesize))
                current_angle -= current_dev_angle
                p1 = polar_to_cart(current_angle+math.pi/2.0, current_linesize/2.0, current_pos)
                p2 = polar_to_cart(current_angle-math.pi/2.0, current_linesize/2.0, current_pos)
                current_edge = [p1, p2]
            else:
                current_pos, current_angle, current_edge, current_linesize = stack.pop()
                current_angle += current_dev_angle
                p1 = polar_to_cart(current_angle+math.pi/2.0, current_linesize/2.0, current_pos)
                p2 = polar_to_cart(current_angle-math.pi/2.0, current_linesize/2.0, current_pos)
                current_edge = [p1, p2]
        else:
            raise
    return img
    
tree = build(iterations, axiom)

def coarse_to_fine_functor(nlevels, ground_arr, render_f):
    """
    creates a function that takes input of unknown variables,
    and outputs the loss
    nlevels: number of levels of this coarse to fine approach
    ground_arr: numpy array, ground truth the optimization tries to achieve
    render_f: a function that takes input of unknown variables, output numpy array for rendered result
    """
    assert nlevels >= 0
    assert ground_arr.shape[0] % (2 ** nlevels) == 0 and ground_arr.shape[1] % (2 ** nlevels) == 0
    def opt(x):
        img = render_f(x)
        diff = ground_arr - img
        loss = numpy.mean(diff ** 2)
        cache = numpy.empty([diff.shape[0]//2, diff.shape[1]//2, 3])
        for n in range(1, nlevels+1):
            scale = 2 ** n
            cache_shape = list(ground_arr.shape)
            cache_shape[0] //= scale
            cache_shape[1] //= scale
            for r in range(cache_shape[0]):
                for c in range(cache_shape[1]):
                    cache[r, c] = numpy.sum(diff[r*scale:(r+1)*scale, c*scale:(c+1)*scale], axis=(0, 1))
            loss += numpy.mean(cache[:cache_shape[0], :cache_shape[1]] ** 2)
        return loss
    return opt
    
def coarse_to_fine_functor_cython(nlevels, ground_arr, f, overlap_loss=True, alpha=1.0):
    """
    creates a function that takes input of unknown variables,
    and outputs the loss
    nlevels: number of levels of this coarse to fine approach
    ground_arr: numpy array, ground truth the optimization tries to achieve
    render_f: a function that takes input of unknown variables, output numpy array for rendered result
    """
    assert nlevels >= 0
    assert ground_arr.shape[0] % (2 ** nlevels) == 0 and ground_arr.shape[1] % (2 ** nlevels) == 0
    img = numpy.empty(size+(3,))
    def opt(x, multiple_loss=False):
        loss = f(img, x)
        diff = ground_arr - numpy.clip(img, 0.0, 1.0)
        if overlap_loss:
            if multiple_loss:
                return alpha * loss, 1e4 * cython_render_diff.all_loss(diff, nlevels)
            else:
                return alpha * loss + 1e4 * cython_render_diff.all_loss(diff, nlevels)
        else:
            return cython_render_diff.all_loss(diff, nlevels)
        #return cython_render_diff.all_loss(diff, nlevels)
    return opt
    
def one_level_loss_functor(level, ground_arr, f):
    img = numpy.empty(size+(3,))
    def opt(x):
        loss = f(img, x)
        diff = ground_arr - numpy.clip(img, 0.0, 1.0)
        return cython_render_diff.loss_level(diff, level)
    return opt

def test1(method):
    ground_arr = render(tree, startpos)
    skimage.io.imsave('test1_ground.png', ground_arr)
    def render_f(img, x):
        return render_prealloc(img, tree, startpos, x[0], x[1])
    opt = coarse_to_fine_functor_cython(9, ground_arr, render_f)
    start = numpy.random.rand(2)
    skimage.io.imsave('test1_start.png', render(tree, startpos, start[0], start[1]))
    ans = scipy.optimize.minimize(opt, start, method=method)
    skimage.io.imsave('test1_out.png', render(tree, startpos, ans['x'][0], ans['x'][1]))
    print(ans)
    
def test2(method, name=''):
    ground_arr = render(tree, startpos, rand=True)
    skimage.io.imsave('py_test2_ground'+name+'.png', numpy.clip(ground_arr, 0.0, 1.0))
    def render_f(img, x):
        return render_prealloc(img, tree, startpos, from_data=x)
    ground_arr = numpy.clip(ground_arr, 0.0, 1.0)
    opt = coarse_to_fine_functor_cython(9, ground_arr, render_f, overlap_loss=False)
    iter = [0]
    def opt_jac(x):
        loss = opt(x)
        jac = scipy.optimize.approx_fprime(x, opt, 1e-8)
        print("iter:", iter[0])
        print("x:", x)
        print("loss:", loss)
        print("jac:", jac)
        for ind in range(len(tree)):
            print(tree[ind], x[ind], jac[ind])
        print()
        iter[0] += 1
        if iter[0] % 50 == 0:
            numpy.save('py_test2'+str(iter[0])+name+'.npy', x)
        return loss, jac
        
    start = numpy.random.rand(len(tree))
    skimage.io.imsave('py_test2_start'+name+'.png', numpy.clip(render(tree, startpos, from_data=start), 0.0, 1.0))
    bounds = []
    for i in range(len(tree)):
        if tree[i] == '1' or tree[i] == '0':
            bounds.append((0.0, None))
        else:
            bounds.append((None, None))
    ans = scipy.optimize.minimize(opt_jac, start, method=method, bounds=bounds, jac=True)
    print("optimize finished")
    print(ans)
    skimage.io.imsave('py_test2_out'+name+'.png', numpy.clip(render(tree, startpos, from_data=ans['x']), 0.0, 1.0))
    return
    best_loss = 1e16
    x = start
    best_ans = None
    for i in range(random_restarts):
        ans = scipy.optimize.minimize(opt, start, method=method, bounds=bounds)
        print("optimize finished")
        if ans['fun'] < best_loss:
            best_loss = ans['fun']
            x = ans['x']
            best_ans = ans
    skimage.io.imsave('py_test2_out.png', numpy.clip(render(tree, startpos, from_data=x), 0.0, 1.0))
    print(best_ans)
 
#test2('L-BFGS-B', 'loss_seperate_test')

def test2_each_level(name=''):
    ground_arr = render(tree, startpos, rand=True)
    skimage.io.imsave('py_test2_ground'+name+'.png', numpy.clip(ground_arr, 0.0, 1.0))
    def render_f(img, x):
        return render_prealloc(img, tree, startpos, from_data=x)
    ground_arr = numpy.clip(ground_arr, 0.0, 1.0)
    start = numpy.random.rand(len(tree))
    skimage.io.imsave('py_test2_start'+name+'.png', numpy.clip(render(tree, startpos, from_data=start), 0.0, 1.0))
    bounds = []
    for i in range(len(tree)):
        if tree[i] == '1' or tree[i] == '0':
            bounds.append((0.0, None))
        else:
            bounds.append((None, None))
    for i in range(9, -1, -1):
        opt = one_level_loss_functor(i, ground_arr, render_f)
        iter = [0]
        def opt_jac(x):
            loss = opt(x)
            jac = scipy.optimize.approx_fprime(x, opt, 1e-8)
            print("iter:", iter[0])
            print("x:", x)
            print("loss:", loss)
            print("jac:", jac)
            for ind in range(len(tree)):
                print(tree[ind], x[ind], jac[ind])
            print()
            iter[0] += 1
            if iter[0] % 10 == 0:
                numpy.save('py_test2'+str(i)+str(iter[0])+name+'.npy', x)
            return loss, jac
        
        ans = scipy.optimize.minimize(opt_jac, start, method='L-BFGS-B', bounds=bounds, jac=True, options={'maxiter': 10})
        start = ans['x']
        numpy.save('py_test2_intermediate'+str(i)+name+'.npy', start)
        skimage.io.imsave('py_test2_intermediate'+name+str(i)+'.png', numpy.clip(render(tree, startpos, from_data=start), 0.0, 1.0))
    print("optimize finished")
    print(ans)
    skimage.io.imsave('py_test2_out'+name+'.png', numpy.clip(render(tree, startpos, from_data=ans['x']), 0.0, 1.0))
    return
    
#test2_each_level('seperate_loss')
 
def test3():
    current_pos = (512, 512)
    for i in range(1):
        img = numpy.zeros((1024, 1024, 3))
        #current_angle = 2.0 * i * math.pi / 10.0 - math.pi
        current_angle = 0.53244265611875985
        current_pos = (350.58684614962084, 671.45162432834286)
        new_pos = polar_to_cart(current_angle, 100.0, current_pos)
        p1 = polar_to_cart(current_angle+math.pi/2.0, 19.291701310097995, current_pos)
        p2 = polar_to_cart(current_angle-math.pi/2.0, 19.291701310097995, current_pos)
        p3 = polar_to_cart(current_angle+math.pi/2.0, 19.291701310097995, new_pos)
        p4 = polar_to_cart(current_angle-math.pi/2.0, 19.291701310097995, new_pos)
        points = numpy.array([p1, p2, p4, p3, p1])
        draw_aapolygon(img, points, b_color)
        skimage.io.imsave('test_'+str(i)+'.png', img)
        
#test3()

def test4(method='L-BFGS-B', prefix='', synthetic_ground=True, var_len=False):
    if synthetic_ground:
        ground_arr = numpy.empty(size+(3,))
        loss = render_overlap_loss(ground_arr, tree, startpos, rand=True, var_len=var_len)
    else:
        ground_arr = skimage.img_as_float(skimage.io.imread('eroded.png'))
    skimage.io.imsave('py_test4_ground'+prefix+'.png', numpy.clip(ground_arr, 0.0, 1.0))
    def render_f(img, x):
        return render_overlap_loss(img, tree, startpos, from_data=x, var_len=var_len)
    opt = coarse_to_fine_functor_cython(9, ground_arr, render_f, overlap_loss=True)
    iter = [0]
    def opt_jac(x):
        loss1, loss2 = opt(x, True)
        loss = loss1 + loss2
        jac = scipy.optimize.approx_fprime(x, opt, 1e-8)
        print("iter:", iter[0])
        print("x:", x)
        print("loss1:", loss1, "loss2:", loss2)
        print("jac:", jac)
        print()
        iter[0] += 1
        if iter[0] % 50 == 0:
            numpy.save('py_test4'+str(iter[0])+prefix+'.npy', x)
        return loss, jac
    
    if var_len:
        start = numpy.random.rand(len(tree)+sum([1 if cha in ['1', '0'] else 0 for cha in tree]))
    else:
        start = numpy.random.rand(len(tree))
    start_img = numpy.empty(size+(3,))
    render_overlap_loss(start_img, tree, startpos, from_data=start, var_len=var_len)
    skimage.io.imsave('py_test4_start'+prefix+'.png', numpy.clip(start_img, 0.0, 1.0))
    bounds = []
    for i in range(start.shape[0]):
        if i >= len(tree):
            bounds.append((0.0, None))
        elif tree[i] == '1' or tree[i] == '0':
            bounds.append((0.0, None))
        else:
            bounds.append((None, None))
    ans = scipy.optimize.minimize(opt_jac, start, method=method, bounds=bounds, jac=True)
    print("optimize finished")
    print(ans)
    out_img = numpy.empty(size+(3,))
    render_overlap_loss(out_img, tree, startpos, from_data=ans['x'], var_len=var_len)
    skimage.io.imsave('py_test4_out'+prefix+'.png', numpy.clip(out_img, 0.0, 1.0))

#test4()

def timing_test_draw():
    img1 = numpy.zeros((1024, 1024, 3))
    img2 = numpy.zeros((1024, 1024, 3))
    img3 = numpy.zeros((1024, 1024, 3))
    img4 = numpy.zeros((1024, 1024, 3))
    points = numpy.array([[20.0, 20.0], [1000.0, 100.0], [100.0, 1000.0], [20.0, 20.0]])
    color = numpy.array([1.0, 1.0, 1.0])
    T1 = time.time()
    draw_aapolygon(img1, points, color)
    T2 = time.time()
    skimage.io.imsave('ground.png', img1)
    print("python time:", T2 - T1)
    T3 = time.time()
    cython_render_diff.draw_aapolygon(img2, points, color)
    T4 = time.time()
    print("cython time:", T4 - T3)
    print(numpy.sum(img2 - img1))
    T5 = time.time()
    cython_render_diff.draw_aapolygon_parallel(img3, points, color)
    T6 = time.time()
    print("cython parallel time:", T6 - T5)
    print(numpy.sum(img3 - img1))

def timing_test_loss():
    ground_arr = render(tree, startpos, 40, math.pi / 4.0)
    def render_f(x):
        return render(tree, startpos, x[0], x[1])
    opt1 = coarse_to_fine_functor(9, ground_arr, render_f)
    T1 = time.time()
    loss1 = opt1([30, math.pi / 4.0])
    T2 = time.time()
    print("python time:", T2 - T1)
    print("python loss:", loss1)
    opt2 = coarse_to_fine_functor_cython(9, ground_arr, render_f)
    T3 = time.time()
    loss2 = opt2([30, math.pi / 4.0])
    T4 = time.time()
    print("cython time:", T4 - T3)
    print("cython loss:", loss2)

def timing_prealloc():
    img = numpy.zeros((1024, 1024, 3))
    T1 = time.time()
    for i in range(100):
        ans = render(tree, startpos, 40, math.pi / 4.0)
    T2 = time.time()
    print("no prealloc time:", T2 - T1)
    T3 = time.time()
    for i in range(100):
        render_prealloc(img, tree, startpos, 40, math.pi / 4.0)
    T4 = time.time()
    print("prealloc time:", T4 - T3)
    skimage.io.imsave('test_no_prealloc.png', ans)
    skimage.io.imsave('test_prealloc.png', img)
    
def timing_overlap():
    img = numpy.empty(size+(3,))
    T1 = time.time()
    for i in range(100):
        render_overlap_loss(img, tree, startpos)
    T2 = time.time()
    print("overlap time:", T2 - T1)
    skimage.io.imsave('test_overlap_loss.png', numpy.clip(img, 0.0, 1.0))
    T3 = time.time()
    for i in range(100):
        render_prealloc(img, tree, startpos)
    T4 = time.time()
    print("non-overlap time:", T4 - T3)
    skimage.io.imsave('test_baseline.png', numpy.clip(img, 0.0, 1.0))
    
#timing_overlap()
    
def main():
    args = sys.argv[1:]
    test4(args[0], args[1], eval(args[2]), eval(args[3]))
    
if __name__ == '__main__':
    #main()
    pass