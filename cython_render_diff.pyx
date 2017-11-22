import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport floor, ceil
cimport cython.parallel

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void draw_aapolygon(np.ndarray[np.float64_t, ndim=3] img,
                          np.ndarray[np.float64_t, ndim=2] points,
                          np.ndarray[np.float64_t, ndim=1] color):
                          
    cdef int bbox_xmax = int(ceil(np.max(points[:, 0])))
    cdef int bbox_xmin = int(floor(np.min(points[:, 0])))
    cdef int bbox_ymax = int(ceil(np.max(points[:, 1])))
    cdef int bbox_ymin = int(floor(np.min(points[:, 1])))
    
    cdef int nlines = points.shape[0] - 1
    
    cdef int i, c
    cdef np.float64_t x0, y0, x1, y1
    cdef np.float64_t dx, dy, dist, line_b
    cdef np.ndarray[np.float64_t, ndim=2] lines = np.empty(
        (nlines, 4))
    
    for i in range(nlines):
        x0 = points[i, 0]
        y0 = points[i, 1]
        x1 = points[i+1, 0]
        y1 = points[i+1, 1]
        dx = x1 - x0
        dy = y1 - y0
        dist = dx ** 2 + dy ** 2
        line_b = dx * y0 - dy * x0
        lines[i, 0] = line_b
        lines[i, 1] = dx
        lines[i, 2] = dy
        lines[i, 3] = dist
    
    cdef int xx, yy
    cdef np.float64_t x, y
    cdef np.float64_t k, ux, uy
    cdef np.float64_t dist_sign, min_dist, current_dist, alpha, bg
    
    for xx in range(bbox_xmin-1, bbox_xmax+2):
        for yy in range(bbox_ymin-1, bbox_ymax+2):
            x = xx + 0.5
            y = yy + 0.5
            min_dist = 1e8
            for i in range(nlines):
                k = ((x - points[i, 0]) * lines[i, 1] + 
                     (y - points[i, 1]) * lines[i, 2]) / lines[i, 3]
                if k < 0:
                    k = 0
                if k >= 1:
                    k = 1
                ux = points[i, 0] + k * lines[i, 1] - x
                uy = points[i, 1] + k * lines[i, 2] - y
                if lines[i, 1] * y - lines[i, 2] * x - lines[i, 0] > 0:
                    dist_sign = 1
                else:
                    dist_sign = -1
                current_dist = dist_sign * (ux ** 2 + uy ** 2) ** 0.5
                if current_dist < min_dist:
                    min_dist = current_dist
            min_dist = get_min_dist(points, lines, x, y)
            if min_dist >= 0:
                for c in range(3):
                    img[xx, yy, c] += color[c]
            elif min_dist > -1:
                alpha = min_dist + 1
                for c in range(3):
                    bg = img[xx, yy, c]
                    #img[xx, yy, c] = alpha * color[c] + (1.0 - alpha) * bg
                    img[xx, yy, c] += alpha * color[c]
    return

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void draw_aapolygon_parallel(np.ndarray[np.float64_t, ndim=3] img,
                                   np.ndarray[np.float64_t, ndim=2] points,
                                   np.ndarray[np.float64_t, ndim=1] color):
                          
    cdef int bbox_xmax = int(ceil(np.max(points[:, 0])))
    cdef int bbox_xmin = int(floor(np.min(points[:, 0])))
    cdef int bbox_ymax = int(ceil(np.max(points[:, 1])))
    cdef int bbox_ymin = int(floor(np.min(points[:, 1])))
    
    cdef int nlines = points.shape[0] - 1
    
    cdef int i, c
    cdef np.float64_t x0, y0, x1, y1
    cdef np.float64_t dx, dy, dist, line_b
    cdef np.ndarray[np.float64_t, ndim=2] lines = np.empty(
        (nlines, 4))
    
    for i in range(nlines):
        x0 = points[i, 0]
        y0 = points[i, 1]
        x1 = points[i+1, 0]
        y1 = points[i+1, 1]
        dx = x1 - x0
        dy = y1 - y0
        dist = dx ** 2 + dy ** 2
        line_b = dx * y0 - dy * x0
        lines[i, 0] = line_b
        lines[i, 1] = dx
        lines[i, 2] = dy
        lines[i, 3] = dist
    
    cdef int xx, yy
    cdef np.float64_t x, y
    cdef np.float64_t k, ux, uy
    cdef np.float64_t dist_sign, min_dist, current_dist, alpha, bg
    
    for xx in cython.parallel.prange(bbox_xmin-1, bbox_xmax+2, nogil=True):
        for yy in range(bbox_ymin-1, bbox_ymax+2):
            x = xx + 0.5
            y = yy + 0.5
            min_dist = 1e8
            for i in range(nlines):
                k = ((x - points[i, 0]) * lines[i, 1] + 
                     (y - points[i, 1]) * lines[i, 2]) / lines[i, 3]
                if k < 0:
                    k = 0
                if k >= 1:
                    k = 1
                ux = points[i, 0] + k * lines[i, 1] - x
                uy = points[i, 1] + k * lines[i, 2] - y
                if lines[i, 1] * y - lines[i, 2] * x - lines[i, 0] > 0:
                    dist_sign = 1
                else:
                    dist_sign = -1
                current_dist = dist_sign * (ux ** 2 + uy ** 2) ** 0.5
                if current_dist < min_dist:
                    min_dist = current_dist
            if min_dist >= 0:
                for c in range(3):
                    img[xx, yy, c] = color[c]
            elif min_dist > -1:
                alpha = min_dist + 1
                for c in range(3):
                    bg = img[xx, yy, c]
                    img[xx, yy, c] = alpha * color[c] + (1.0 - alpha) * bg
    return
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.float64_t draw_aapolygon_overlap_loss(np.ndarray[np.float64_t, ndim=3] img,
                                               np.ndarray[np.float64_t, ndim=2] points1,
                                               np.ndarray[np.float64_t, ndim=2] points2,
                                               np.ndarray[np.float64_t, ndim=1] color):
    
    cdef int nlines1 = points1.shape[0] - 1
    cdef int nlines2 = points2.shape[0] - 1
    
    cdef np.float64_t loss
    loss = 0.0
    
    cdef int i, c
    cdef np.float64_t x0, y0, x1, y1
    cdef np.float64_t dx, dy, dist, line_b
    cdef np.ndarray[np.float64_t, ndim=2] lines1 = np.empty((nlines1, 4))
    cdef np.ndarray[np.float64_t, ndim=2] lines2 = np.empty((nlines2, 4))
    
    for i in range(nlines1):
        x0 = points1[i, 0]
        y0 = points1[i, 1]
        x1 = points1[i+1, 0]
        y1 = points1[i+1, 1]
        dx = x1 - x0
        dy = y1 - y0
        dist = dx ** 2 + dy ** 2
        line_b = dx * y0 - dy * x0
        lines1[i, 0] = line_b
        lines1[i, 1] = dx
        lines1[i, 2] = dy
        lines1[i, 3] = dist
        
    for i in range(nlines2):
        x0 = points2[i, 0]
        y0 = points2[i, 1]
        x1 = points2[i+1, 0]
        y1 = points2[i+1, 1]
        dx = x1 - x0
        dy = y1 - y0
        dist = dx ** 2 + dy ** 2
        line_b = dx * y0 - dy * x0
        lines2[i, 0] = line_b
        lines2[i, 1] = dx
        lines2[i, 2] = dy
        lines2[i, 3] = dist
    
    cdef int xx, yy
    cdef np.float64_t x, y
    cdef np.float64_t min_dist1, min_dist2, min_dist, overlap, alpha, bg
    
    cdef int bbox_xmax1 = int(ceil(np.max(points1[:, 0])))
    cdef int bbox_xmin1 = int(floor(np.min(points1[:, 0])))
    cdef int bbox_ymax1 = int(ceil(np.max(points1[:, 1])))
    cdef int bbox_ymin1 = int(floor(np.min(points1[:, 1])))
    
    cdef int bbox_xmax2 = int(ceil(np.max(points2[:, 0])))
    cdef int bbox_xmin2 = int(floor(np.min(points2[:, 0])))
    cdef int bbox_ymax2 = int(ceil(np.max(points2[:, 1])))
    cdef int bbox_ymin2 = int(floor(np.min(points2[:, 1])))
    
    cdef int bbox_xmax, bbox_xmin, bbox_ymax, bbox_ymin
    bbox_xmax = np.min([bbox_xmax1, bbox_xmax2])
    bbox_xmin = np.max([bbox_xmin1, bbox_xmin2])
    bbox_ymax = np.min([bbox_ymax1, bbox_ymax2])
    bbox_ymin = np.max([bbox_ymin1, bbox_ymin2])
    
    for xx in range(bbox_xmin-1, bbox_xmax+2):
        for yy in range(bbox_ymin-1, bbox_ymax+2):
            x = xx + 0.5
            y = yy + 0.5
            min_dist1 = get_min_dist(points1, lines1, x, y)
            min_dist2 = get_min_dist(points2, lines2, x, y)
            if min_dist1 < min_dist2:
                min_dist = min_dist2
                overlap = min_dist1
            else:
                min_dist = min_dist1
                overlap = min_dist2
            if min_dist >= 0:
                for c in range(3):
                    img[xx, yy, c] = color[c]
            elif min_dist > -1:
                alpha = min_dist + 1
                for c in range(3):
                    bg = img[xx, yy, c]
                    img[xx, yy, c] = alpha * color[c] + (1.0 - alpha) * bg
                    #img[xx, yy, c] += alpha * color[c]
            if overlap > -1:
                if overlap < 1.0:
                    loss += overlap + 1
                else:
                    loss += 1.0
                
    cdef int use_p1
    cdef int xl, xh, yl, yh
    cdef int case
    
    for case in range(4):
        if case == 0:
            if bbox_xmin1 < bbox_xmin2:
                xl = bbox_xmin1 - 1
                yl = bbox_ymin1 - 1
                yh = bbox_ymax1 + 2
                use_p1 = 1
            else:
                xl = bbox_xmin2 - 1
                yl = bbox_ymin2 - 1
                yh = bbox_ymax2 + 2
                use_p1 = 0
            xh = bbox_xmin - 1
        elif case == 1:
            if bbox_xmax1 > bbox_xmax2:
                xh = bbox_xmax1 + 2
                yl = bbox_ymin1 - 1
                yh = bbox_ymax1 + 2
                use_p1 = 1
            else:
                xh = bbox_xmax2 + 2
                yl = bbox_ymin2 - 1
                yh = bbox_ymax2 + 2
                use_p1 = 0
            xl = bbox_xmax + 2
        elif case == 2:
            if bbox_ymin1 < bbox_ymin2:
                yl = bbox_ymin1 - 1
                xl = bbox_xmin - 1
                xh = bbox_xmax + 2
                #xl = bbox_xmin1 - 1
                #xh = bbox_xmax1 + 2
                use_p1 = 1
            else:
                yl = bbox_ymin2 - 1
                xl = bbox_xmin - 1
                xh = bbox_xmax + 2
                #xl = bbox_xmin2 - 1
                #xh = bbox_xmax2 + 2
                use_p1 = 0
            yh = bbox_ymin - 1
        else:
            if bbox_ymax1 > bbox_ymax2:
                yh = bbox_ymax1 + 2
                xl = bbox_xmin - 1
                xh = bbox_xmax + 2
                #xl = bbox_xmin1 - 1
                #xh = bbox_xmax1 + 2
                use_p1 = 1
            else:
                yh = bbox_ymax2 + 2
                xl = bbox_xmin - 1
                xh = bbox_xmax + 2
                #xl = bbox_xmin2 - 1
                #xh = bbox_xmax2 + 2
                use_p1 = 0
            yl = bbox_ymax + 2
    
        for xx in range(xl, xh):
            for yy in range(yl, yh):
                x = xx + 0.5
                y = yy + 0.5
                if use_p1 == 1:
                    min_dist = get_min_dist(points1, lines1, x, y)
                else:
                    min_dist = get_min_dist(points2, lines2, x, y)
                if min_dist >= 0:
                    for c in range(3):
                        img[xx, yy, c] = color[c]
                elif min_dist > -1:
                    alpha = min_dist + 1
                    for c in range(3):
                        bg = img[xx, yy, c]
                        img[xx, yy, c] = alpha * color[c] + (1.0 - alpha) * bg
                        #img[xx, yy, c] += alpha * color[c]
                
    return loss
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.float64_t get_min_dist(np.ndarray[np.float64_t, ndim=2] points,
                                np.ndarray[np.float64_t, ndim=2] lines,
                                np.float64_t x,
                                np.float64_t y):
    cdef int i
    cdef np.float64_t k, ux, uy
    cdef np.float64_t dist_sign, min_dist, current_dist
    cdef int nlines = points.shape[0] - 1
    min_dist = 1e8
    for i in range(nlines):
        k = ((x - points[i, 0]) * lines[i, 1] + 
             (y - points[i, 1]) * lines[i, 2]) / lines[i, 3]
        if k < 0:
            k = 0
        if k >= 1:
            k = 1
        ux = points[i, 0] + k * lines[i, 1] - x
        uy = points[i, 1] + k * lines[i, 2] - y
        if lines[i, 1] * y - lines[i, 2] * x - lines[i, 0] > 0:
            dist_sign = 1
        else:
            dist_sign = -1
        current_dist = dist_sign * (ux ** 2 + uy ** 2) ** 0.5
        if current_dist < min_dist:
            min_dist = current_dist
    return min_dist
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.float64_t level_loss(np.ndarray[np.float64_t, ndim=3] diff,
                                       int level):

    cdef int h = diff.shape[0]
    cdef int w = diff.shape[1]
    cdef int i, j, c
    cdef np.float64_t loss = 0.0
    cdef int scale = 2 ** level
    cdef np.ndarray[np.float64_t, ndim=3] accum = np.zeros((h//scale, w//scale, 3))
    
    for i in range(accum.shape[0]):
        for j in range(accum.shape[1]):
            for c in range(3):
                accum[i, j, c] = np.sum(diff[i*scale:(i+1)*scale, j*scale:(j+1)*scale, c])
    return np.mean(accum ** 2)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.float64_t all_loss(np.ndarray[np.float64_t, ndim=3] diff,
                            int nlevels):
    cdef np.float64_t loss = np.mean(diff ** 2.0)
    cdef int n, r, c, xx, yy, rr, cc
    cdef int scale, h, w
    cdef np.ndarray[np.float64_t, ndim=1] accum = np.empty(3)
    cdef np.float64_t current_loss
    cdef int diff_w, diff_h
    diff_h = diff.shape[0]
    diff_w = diff.shape[1]
    
    for n in range(1, nlevels+1):
        scale = 2 ** n
        h = diff_h // scale
        w = diff_w // scale
        current_loss = 0.0
        
        for rr in range(h):
            for cc in range(w):
                r = scale * rr
                c = scale * cc
        #for r in range(0, diff_h, scale):
        #    for c in range(0, diff_w, scale):
                accum[0] = 0.0
                accum[1] = 0.0
                accum[2] = 0.0
                for xx in range(scale):
                    for yy in range(scale):
                        accum[0] += diff[r+xx, c+yy, 0]
                        accum[1] += diff[r+xx, c+yy, 1]
                        accum[2] += diff[r+xx, c+yy, 2]
                current_loss += accum[0]**2 + accum[1]**2 + accum[2]**2
        # correct scale
        loss += current_loss / (w * h * 3.0) ** 2
    return loss

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)    
cpdef np.float64_t loss_level(np.ndarray[np.float64_t, ndim=3] diff,
                              int level):
    cdef np.float64_t loss = 0.0
    cdef int r, c, scale, h, w, xx, yy, rr, cc
    cdef np.ndarray[np.float64_t, ndim=1] accum = np.empty(3)
    cdef int diff_w, diff_h
    diff_h = diff.shape[0]
    diff_w = diff.shape[1]
    
    scale = 2 ** level
    h = diff_h // scale
    w = diff_w // scale
    for rr in range(h):
        for cc in range(w):
            r = scale * rr
            c = scale * cc
            accum[0] = 0.0
            accum[1] = 0.0
            accum[2] = 0.0
            for xx in range(scale):
                for yy in range(scale):
                    accum[0] += diff[r+xx, c+yy, 0]
                    accum[1] += diff[r+xx, c+yy, 1]
                    accum[2] += diff[r+xx, c+yy, 2]
            loss += accum[0] ** 2 + accum[1] ** 2 + accum[2] ** 2
    return loss / (w * h * 3.0)