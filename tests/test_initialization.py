import numpy as np

from uav_mfg.initialization import intersects_box


def test_segment_box_intersection_blocks_corner_cut():
    start=np.array([[-1.,0.5,0.5],[-1.,2.,0.5],[-1.,-1.,0.5]])
    end=np.array([[2.,0.5,0.5],[2.,2.,0.5],[2.,2.,0.5]])
    np.testing.assert_array_equal(intersects_box(start,end,[0,0,0],[1,1,1]),[True,False,True])


def test_parallel_segment_outside_box_is_not_blocked():
    np.testing.assert_array_equal(intersects_box(np.array([[2.,-1.,0.5]]),np.array([[2.,2.,0.5]]),[0,0,0],[1,1,1]),[False])
