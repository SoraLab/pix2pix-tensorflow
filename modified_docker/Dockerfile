FROM affinelayer/pix2pix-tensorflow

RUN cd $CAFFE_ROOT && \
    sed -i "647s|vir|//vir|" include/caffe/loss_layers.hpp && \
    sed -i "648s|    con|//    con|" include/caffe/loss_layers.hpp && \
    cd build && \
    cmake -DCPU_ONLY=0 -DUSE_CUDNN=0 .. && \
    make -j"$(nproc)"
