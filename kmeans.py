# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 15:57:56 2017

@author: lxy
"""

import array
import struct
import numpy as np
import datetime
import pyopencl as cl
import sys
DIM=2
class_n=0
data_n=0
iteration_n=1024
mf = cl.mem_flags
def kmeans(it_n,class_n,data_n, centroids_x,centroids_y,data_x,data_y,partitioned):
    context = cl.create_some_context()
    queue =  cl.CommandQueue(context);
    for i in range(0,class_n):
        print centroids_x[i],centroids_y[i]
    with open("kernel.cl", 'r') as fin:
        program = cl.Program(context, fin.read()).build()
    assign = program.assign
    buf_centroids_x=cl.Buffer(context,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=centroids_x)
    buf_centroids_y =cl.Buffer(context,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=centroids_y)
    buf_data_x=cl.Buffer(context,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=data_x)
    buf_data_y=cl.Buffer(context,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=data_y)
    buf_parts = cl.Buffer(context,mf.READ_WRITE | mf.COPY_HOST_PTR,hostbuf=partitioned)
    dbl_max =100000.0
    for i in range(0,it_n):
        assign(queue,(data_n,),None,buf_centroids_x,buf_centroids_y,buf_data_x,buf_data_y, buf_parts,np.int32(class_n),np.int32(data_n),np.float32(dbl_max))
        cl.enqueue_barrier(queue)
        e =cl.enqueue_copy(queue,partitioned,buf_parts)
        e.wait()
        count = np.zeros(class_n).astype(np.int32)
        for i in range(0,class_n):
            centroids_x[i]=0.0
            centroids_y[i]=0.0
        for i in range(0,data_n):
            centroids_x[partitioned[i]] +=data_x[i]
            centroids_y[partitioned[i]] +=data_y[i]
            count[partitioned[i]]+=1
        for i in range(0,class_n):
            if count[i] != 0:
                centroids_x[i] /= count[i]
                centroids_y[i] /= count[i]
        buf_centroids_x=cl.Buffer(context,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=centroids_x)
        buf_centroids_y =cl.Buffer(context,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=centroids_y)
    print partitioned
    return partitioned
    
def transpoint(data):
    x=[]
    y=[]
    for i in range(0,len(data)/2):
        x.append(data[i*2])
        y.append(data[i*2+1])
    return np.array(x),np.array(y)
if __name__=='__main__':
    if len(sys.argv) < 4:
        print('{0} <centroid file> <data file> <result file>'.format(sys.argv[0]))
        sys.exit()
    centfile=sys.argv[1]
    print centfile
    datafile = sys.argv[2]
    resultfile = sys.argv[3]
    with open(centfile, 'rb') as input_f:
        size = struct.unpack('I', input_f.read(struct.calcsize('I')))[0]
        centroids = array.array('f')
        centroids.fromfile(input_f, size * DIM)
        class_n =len(centroids)/2
        centroids=np.array(centroids).astype(np.float32)
    with open(datafile, 'rb') as input_f:
        size = struct.unpack('I', input_f.read(struct.calcsize('I')))[0]
        data = array.array('f')
        data.fromfile(input_f, size * DIM)
        data_n = len(data)/2
        data = np.array(data).astype(np.float32)
    iteration_n= 1024
    
    partitioned = np.zeros(data_n).astype(np.int32);
    centroids_x,centroids_y = transpoint(centroids)
    data_x,data_y =transpoint(data)
    begin = datetime.datetime.now()
    
    result=kmeans(iteration_n,class_n,data_n,centroids_x,centroids_y,data_x,data_y,partitioned)
    end = datetime.datetime.now()
    #result.tofile(resultfile)
    output_f = open(resultfile, 'wb')
    output_f.write(struct.pack('I', len(result)))
    result.tofile(output_f)
    output_f.close()
    
    print end-begin

