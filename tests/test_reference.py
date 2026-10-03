import numpy as np

from uav_mfg.reference import local_eikonal_update


def test_one_dimensional_exact_update():
    assert local_eikonal_update([2,np.inf,np.inf],[1,1,1],5)==2.2


def test_three_dimensional_radial_causal_update():
    result=local_eikonal_update([0,0,0],[1,1,1],1)
    np.testing.assert_allclose(result,1/np.sqrt(3))


def test_inactive_axis_does_not_change_characteristic():
    result=local_eikonal_update([0,100,np.inf],[2,1,1],2)
    assert result==1
