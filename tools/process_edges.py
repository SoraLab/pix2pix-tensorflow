from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse
import os
import tempfile
import subprocess
import numpy as np
import threading
import time
import matplotlib.image as mpimg


parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", required=True,
                    help="path to folder containing images")
parser.add_argument("--output_dir", required=True, help="output path")
a = parser.parse_args()


net = None


def run_caffe(src):
    # lazy load caffe and create net
    global net
    if net is None:
        # don't require caffe unless we are doing edge detection
        os.environ["GLOG_minloglevel"] = "2"  # disable logging from caffe
        import caffe
        caffe.set_mode_gpu()
        # using this requires using the docker image or assembling a bunch of dependencies
        # and then changing these hardcoded paths
        net = caffe.Net("/opt/caffe/examples/hed/deploy.prototxt",
                        "/opt/caffe/hed_pretrained_bsds.caffemodel", caffe.TEST)

    net.blobs["data"].reshape(1, *src.shape)
    net.blobs["data"].data[...] = src
    net.forward()
    return net.blobs["sigmoid-fuse"].data[0][0, :, :]


def edges(src):
    # based on https://github.com/phillipi/pix2pix/blob/master/scripts/edges/batch_hed.py
    # and https://github.com/phillipi/pix2pix/blob/master/scripts/edges/PostprocessHED.m
    import scipy.io
    src = src * 255
    border = 128  # put a padding around images since edge detection seems to detect edge of image
    src = src[:, :, :3]  # remove alpha channel if present
    src = np.pad(src, ((border, border), (border, border), (0, 0)), "reflect")
    src = src[:, :, ::-1]
    src -= np.array((104.00698793, 116.66876762, 122.67891434))
    src = src.transpose((2, 0, 1))

    # [height, width, channels] => [batch, channel, height, width]
    fuse = run_caffe(src)
    fuse = fuse[border:-border, border:-border]

    with tempfile.NamedTemporaryFile(suffix=".png") as png_file, tempfile.NamedTemporaryFile(suffix=".mat") as mat_file:
        scipy.io.savemat(mat_file.name, {"input": fuse})

        octave_code = r"""
E = 1-load(input_path).input;
E = imresize(E, [image_width,image_width]);
E = 1 - E;
E = single(E);
[Ox, Oy] = gradient(convTri(E, 4), 1);
[Oxx, ~] = gradient(Ox, 1);
[Oxy, Oyy] = gradient(Oy, 1);
O = mod(atan(Oyy .* sign(-Oxy) ./ (Oxx + 1e-5)), pi);
E = edgesNmsMex(E, O, 1, 5, 1.01, 1);
E = double(E >= max(eps, threshold));
E = bwmorph(E, 'thin', inf);
E = bwareaopen(E, small_edge);
E = 1 - E;
E = uint8(E * 255);
imwrite(E, output_path);
"""

        config = dict(
            input_path="'%s'" % mat_file.name,
            output_path="'%s'" % png_file.name,
            image_width=256,
            threshold=25.0 / 255.0,
            small_edge=5,
        )

        args = ["octave"]
        for k, v in config.items():
            args.extend(["--eval", "%s=%s;" % (k, v)])

        args.extend(["--eval", octave_code])
        try:
            subprocess.check_output(args, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            print("octave failed")
            print("returncode:", e.returncode)
            print("output:", e.output)
            raise
        return mpimg.imread(png_file.name)


def process(src_path, dst_path):
    src = mpimg.imread(src_path)
    dst = edges(src)
    mpimg.imsave(dst_path, dst, cmap='gray')


complete_lock = threading.Lock()
start = None
num_complete = 0
total = 0


def complete():
    global num_complete, rate, last_complete

    with complete_lock:
        num_complete += 1
        now = time.time()
        elapsed = now - start
        rate = num_complete / elapsed
        if rate > 0:
            remaining = (total - num_complete) / rate
        else:
            remaining = 0

        print("%d/%d complete  %0.2f images/sec  %dm%ds elapsed  %dm%ds remaining" %
              (num_complete, total, rate, elapsed // 60, elapsed % 60, remaining // 60, remaining % 60))

        last_complete = now


def main():
    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    src_paths = []
    dst_paths = []

    skipped = 0
    # for src_path in im.find(a.input_dir):
    for src_path in os.listdir(a.input_dir):
        name, _ = os.path.splitext(os.path.basename(src_path))
        dst_path = os.path.join(a.output_dir, name + ".png")
        if os.path.exists(dst_path):
            skipped += 1
        else:
            src_paths.append(os.path.join(a.input_dir, src_path))
            dst_paths.append(dst_path)

    print("skipping %d files that already exist" % skipped)

    global total
    total = len(src_paths)

    print("processing %d files" % total)

    global start
    start = time.time()

    for src_path, dst_path in zip(src_paths, dst_paths):
        process(src_path, dst_path)
        complete()
    net = None


main()
