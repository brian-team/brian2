'''
An example that uses a function from external C library (OpenCV in this case).
Works for all C-based code generation targets (i.e. for weave and cpp_standalone
device) and for numpy (using the Python bindings).

This example needs a working installation of OpenCV2 and its Python bindings.
It has been tested on Ubuntu 14.04 with OpenCV 2.4.8 (libopencv-dev and
python-opencv packages).
'''
import os
import urllib2
import cv2  # Import OpenCV2
import cv2.cv as cv  # Import the cv subpackage, needed for some constants

from brian2 import *

defaultclock.dt = 1*ms
prefs.codegen.target = 'weave'
prefs.logging.std_redirection = False
set_device('cpp_standalone')
filename = os.path.abspath('Megamind.avi')

if not os.path.exists(filename):
    print('Downloading the example video file')
    response = urllib2.urlopen('http://docs.opencv.org/2.4/_downloads/Megamind.avi')
    data = response.read()
    with open(filename, 'wb') as f:
        f.write(data)

video = cv2.VideoCapture(filename)
width, height, frame_count = (int(video.get(cv.CV_CAP_PROP_FRAME_WIDTH)),
                              int(video.get(cv.CV_CAP_PROP_FRAME_HEIGHT)),
                              int(video.get(cv.CV_CAP_PROP_FRAME_COUNT)))
fps = 24
time_between_frames = 1*second/fps

# Links the necessary libraries
prefs.codegen.cpp.libraries += ['opencv_core',
                                'opencv_highgui']

# Includes the header files in all generated files
prefs.codegen.cpp.headers += ['<opencv2/core/core.hpp>',
                              '<opencv2/highgui/highgui.hpp>']

# Pass in values as macros
# Note that in general we could also pass in the filename this way, but to get
# the string quoting right is unfortunately quite difficult
prefs.codegen.cpp.define_macros += [('VIDEO_WIDTH', width),
                                    ('VIDEO_HEIGHT', height)]
@implementation('cpp', '''
double* get_frame(bool new_frame)
{
    // The following initializations will only be executed once
    static cv::VideoCapture source("VIDEO_FILENAME");
    static cv::Mat frame;
    static double* grayscale_frame = (double*)malloc(VIDEO_WIDTH*VIDEO_HEIGHT*sizeof(double));
    if (new_frame)
    {
        source >> frame;
        double mean_value = 0;
        for (int row=0; row<VIDEO_HEIGHT; row++)
            for (int col=0; col<VIDEO_WIDTH; col++)
            {
                const double grayscale_value = (frame.at<cv::Vec3b>(row, col)[0] +
                                                frame.at<cv::Vec3b>(row, col)[1] +
                                                frame.at<cv::Vec3b>(row, col)[2])/(3.0*128);
                mean_value += grayscale_value / (VIDEO_WIDTH * VIDEO_HEIGHT);
                grayscale_frame[row*VIDEO_WIDTH + col] = grayscale_value;
            }
        // subtract the mean
        for (int i=0; i<VIDEO_HEIGHT*VIDEO_WIDTH; i++)
            grayscale_frame[i] -= mean_value;
    }
    return grayscale_frame;
}

double video_input(const int x, const int y)
{
    // Get the current frame (or a new frame in case we are asked for the first
    // element
    double *frame = get_frame(x==0 && y==0);
    return frame[y*VIDEO_WIDTH + x];
}
'''.replace('VIDEO_FILENAME', filename))
@check_units(x=1, y=1, result=1)
def video_input(x, y):
    # we assume this will only be called in the custom operation (and not for
    # example in a reset or synaptic statement), so we don't need to do indexing
    # but we can directly return the full result
    _, frame = video.read()
    grayscale = frame.mean(axis=2)
    grayscale /= 128.  # scale everything between 0 and 2
    return grayscale.ravel() - grayscale.ravel().mean()


N = width * height
tau, tau_th = 10*ms, time_between_frames
G = NeuronGroup(N, '''dv/dt = (-v + I)/tau : 1
                      dv_th/dt = -v_th/tau_th : 1
                      row : integer (constant)
                      column : integer (constant)
                      I : 1 # input current''',
                threshold='v>v_th', reset='v=0; v_th = 3*v_th + 1.0',
                method='linear')
G.v_th = 1
G.row = 'i/width'
G.column = 'i%width'

G.run_regularly('I = video_input(column, row)',
                dt=time_between_frames)
mon = SpikeMonitor(G)
runtime = frame_count*time_between_frames
run(runtime, report='text')
device.build(compile=True, run=True)

# Avoid going through the whole Brian2 indexing machinery too much
i, t, row, column = mon.i[:], mon.t[:], G.row[:], G.column[:]

import matplotlib.animation as animation

# TODO: Use overlapping windows
stepsize = 100*ms
def next_spikes():
    step = next_spikes.step
    if step*stepsize > runtime:
        next_spikes.step=0
        raise StopIteration()
    spikes = i[(t>=step*stepsize) & (t<(step+1)*stepsize)]
    next_spikes.step += 1
    yield column[spikes], row[spikes]
next_spikes.step = 0

fig, ax = plt.subplots()
dots, = ax.plot([], [], 'k.', markersize=2, alpha=.25)
ax.set_xlim(0, width)
ax.set_ylim(0, height)
ax.invert_yaxis()
def run(data):
    x, y = data
    dots.set_data(x, y)

ani = animation.FuncAnimation(fig, run, next_spikes, blit=False, repeat=True,
                              repeat_delay=1000)
plt.show()
