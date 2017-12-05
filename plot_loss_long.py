import numpy
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot

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