from nose.plugins.attrib import attr
from numpy.testing.utils import assert_equal, assert_allclose, assert_raises
import numpy as np

from brian2.spatialneuron import *
from brian2.units import um, second, DimensionMismatchError


@attr('codegen-independent')
def test_attributes_soma():
    soma = Soma(diameter=10*um)
    assert isinstance(soma, Morphology)
    # Single compartment
    assert soma.n == 1
    assert soma.n_sections == 1
    assert len(soma) == 1
    # Compartment attributes
    assert_equal(soma.diameter, [10]*um)
    assert_equal(soma.length, [10]*um)
    assert_equal(soma.distance, [0]*um)
    assert_equal(soma.total_distance, 0*um)
    assert_equal(soma.r_length, 0*um)
    assert_equal(soma.area, np.pi*soma.diameter**2)
    assert_allclose(soma.volume, 1.0/6.0*np.pi*(10*um)**3)

    # No coordinates were specified
    assert soma.start_x is None
    assert soma.start_y is None
    assert soma.start_z is None
    assert soma.x is None
    assert soma.y is None
    assert soma.z is None
    assert soma.end_x is None
    assert soma.end_y is None
    assert soma.end_z is None


@attr('codegen-independent')
def test_attributes_soma_coordinates():
    # Specify only one of the coordinates
    xyz = {'x', 'y', 'z'}
    for coord in xyz:
        kwds = {coord: 5*um}
        soma = Soma(diameter=10*um, **kwds)
        # Length shouldn't change (not defined by coordinates but by the diameter)
        assert_equal(soma.length, [10]*um)
        assert_equal(soma.distance, [0]*um)

        # Coordinates should be specified now, with 0 values for the other
        # coordinates
        for other_coord in xyz - {coord}:
            assert_equal(getattr(soma, 'start_' + other_coord), [0]*um)
            assert_equal(getattr(soma, other_coord), [0]*um)
            assert_equal(getattr(soma, 'end_' + other_coord), [0]*um)

        assert_equal(getattr(soma, 'start_' + coord), [5]*um)
        assert_equal(getattr(soma, coord), [5]*um)
        assert_equal(getattr(soma, 'end_' + coord), [5]*um)

    # Specify all coordinates
    soma = Soma(diameter=10*um, x=1*um, y=2*um, z=3*um)
    # Length shouldn't change (not defined by coordinates but by the diameter)
    assert_equal(soma.length, [10]*um)
    assert_equal(soma.distance, [0]*um)

    assert_equal(soma.start_x, 1*um)
    assert_equal(soma.x, 1*um)
    assert_equal(soma.end_x, 1*um)
    assert_equal(soma.start_y, 2*um)
    assert_equal(soma.y, 2*um)
    assert_equal(soma.end_y, 2*um)
    assert_equal(soma.start_z, 3*um)
    assert_equal(soma.z, 3*um)
    assert_equal(soma.end_z, 3*um)


@attr('codegen-independent')
def test_attributes_cylinder():
    n = 10
    cylinder = Cylinder(n, diameter=10*um, length=200*um)
    assert isinstance(cylinder, Morphology)
    # Single section with 10 compartments
    assert cylinder.n == n
    assert cylinder.n_sections == 1
    assert len(cylinder) == n

    # Compartment attributes
    assert_equal(cylinder.diameter, np.ones(n)*10*um)
    assert_equal(cylinder.length, np.ones(n)*20*um)
    assert_equal(cylinder.distance, np.arange(n)*20*um + 10*um)
    assert_equal(cylinder.total_distance, 200*um)
    # TODO: r_length
    assert_allclose(cylinder.area, np.pi*cylinder.diameter*cylinder.length)
    assert_allclose(cylinder.volume, 1.0/4.0*np.pi*cylinder.diameter**2*cylinder.length)

    # No coordinates were specified
    assert cylinder.start_x is None
    assert cylinder.start_y is None
    assert cylinder.start_z is None
    assert cylinder.x is None
    assert cylinder.y is None
    assert cylinder.z is None
    assert cylinder.end_x is None
    assert cylinder.end_y is None
    assert cylinder.end_z is None


@attr('codegen-independent')
def test_attributes_cylinder_coordinates_single():
    # Specify only the end-point of the section
    n = 10
    # Specify only one of the coordinates
    xyz = {'x', 'y', 'z'}
    for coord in xyz:
        kwds = {coord: 200*um}
        cylinder = Cylinder(n, diameter=10*um, **kwds)
        assert_equal(cylinder.diameter, np.ones(n)*10*um)
        assert_equal(cylinder.length, np.ones(n)*20*um)
        assert_equal(cylinder.distance, np.arange(n)*20*um + 10*um)
        assert_equal(cylinder.total_distance, 200*um)

        # Coordinates should be specified now, with 0 values for the other
        # coordinates
        for other_coord in xyz - {coord}:
            assert_equal(getattr(cylinder, 'start_' + other_coord), np.zeros(n)*um)
            assert_equal(getattr(cylinder, other_coord), np.zeros(n)*um)
            assert_equal(getattr(cylinder, 'end_' + other_coord), np.zeros(n)*um)

        assert_equal(getattr(cylinder, 'start_' + coord), np.arange(n)*20*um)
        assert_equal(getattr(cylinder, coord), np.arange(n)*20*um + 10*um)
        assert_equal(getattr(cylinder, 'end_' + coord), np.arange(n)*20*um + 20*um)

    # Specify all coordinates
    val = 200.0/np.sqrt(3.0)*um
    cylinder = Cylinder(n, diameter=10*um, x=val, y=val, z=val)

    assert_equal(cylinder.diameter, np.ones(n)*10*um)
    assert_allclose(cylinder.length, np.ones(n)*20*um)
    assert_allclose(cylinder.distance, np.arange(n)*20*um + 10*um)
    assert_allclose(cylinder.total_distance, 200*um)

    for coord in ['x', 'y', 'z']:
        assert_allclose(getattr(cylinder, 'start_' + coord), np.arange(n)*val/n)
        assert_allclose(getattr(cylinder, coord), np.arange(n)*val/n + 0.5*val/n)
        assert_allclose(getattr(cylinder, 'end_' + coord), np.arange(n)*val/n + val/n)


@attr('codegen-independent')
def test_attributes_cylinder_coordinates_endpoints():
    # Specify only the end-point of the section
    n = 3
    # Specify only one of the coordinates
    xyz = {'x', 'y', 'z'}
    for coord in xyz:
        kwds = {coord: [5, 15, 30]*um}
        cylinder = Cylinder(n, diameter=10*um, **kwds)
        assert_equal(cylinder.diameter, np.ones(n)*10*um)
        assert_equal(cylinder.length, [5, 10, 15]*um)
        assert_equal(cylinder.distance, [2.5, 10, 22.5]*um)
        assert_equal(cylinder.total_distance, 30*um)

        # Coordinates should be specified now, with 0 values for the other
        # coordinates
        for other_coord in xyz - {coord}:
            assert_equal(getattr(cylinder, 'start_' + other_coord), np.zeros(n)*um)
            assert_equal(getattr(cylinder, other_coord), np.zeros(n)*um)
            assert_equal(getattr(cylinder, 'end_' + other_coord), np.zeros(n)*um)

        assert_equal(getattr(cylinder, 'start_' + coord), [0, 5, 15]*um)
        assert_equal(getattr(cylinder, coord), [2.5, 10.0, 22.5]*um)
        assert_equal(getattr(cylinder, 'end_' + coord), [5, 15, 30]*um)

    # Specify all coordinates
    cylinder = Cylinder(n, diameter=10*um, x=[1, 1, 1]*um, y=[0, 1, 1]*um,
                        z=[0, 0, 1]*um)

    assert_equal(cylinder.diameter, np.ones(n)*10*um)
    assert_allclose(cylinder.length, np.ones(n)*um)
    assert_allclose(cylinder.distance, np.arange(n)*um + .5*um)
    assert_allclose(cylinder.total_distance, 3*um)

    assert_equal(cylinder.start_x, [0, 1, 1]*um)
    assert_equal(cylinder.x, [0.5, 1, 1]*um)
    assert_equal(cylinder.end_x, [1, 1, 1]*um)
    assert_equal(cylinder.start_y, [0, 0, 1]*um)
    assert_equal(cylinder.y, [0, .5, 1]*um)
    assert_equal(cylinder.end_y, [0, 1, 1]*um)
    assert_equal(cylinder.start_z, [0, 0, 0]*um)
    assert_equal(cylinder.z, [0, 0, 0.5]*um)
    assert_equal(cylinder.end_z, [0, 0, 1]*um)

    # Specify varying diameters
    cylinder = Cylinder(n, diameter=[10, 5, 2.5]*um, x=[1, 1, 1]*um,
                        y=[0, 1, 1]*um, z=[0, 0, 1]*um)
    assert_equal(cylinder.diameter, [10, 5, 2.5]*um)


@attr('codegen-independent')
def test_attributes_cylinder_coordinates_allpoints():
    n = 3
    # Specify all coordinates, including the start point of the section
    cylinder = Cylinder(n, diameter=10*um,
                        x=[10, 11, 11, 11]*um,
                        y=[100, 100, 101, 101]*um,
                        z=[1000, 1000, 1000, 1001]*um)

    assert_equal(cylinder.diameter, np.ones(n)*10*um)
    assert_allclose(cylinder.length, np.ones(n)*um)
    assert_allclose(cylinder.distance, np.arange(n)*um + .5*um)
    assert_allclose(cylinder.total_distance, 3*um)

    assert_allclose(cylinder.start_x, [10, 11, 11]*um)
    assert_allclose(cylinder.x, [10.5, 11, 11]*um)
    assert_allclose(cylinder.end_x, [11, 11, 11]*um)
    assert_allclose(cylinder.start_y, [100, 100, 101]*um)
    assert_allclose(cylinder.y, [100, 100.5, 101]*um)
    assert_allclose(cylinder.end_y, [100, 101, 101]*um)
    assert_allclose(cylinder.start_z, [1000, 1000, 1000]*um)
    assert_allclose(cylinder.z, [1000, 1000, 1000.5]*um)
    assert_allclose(cylinder.end_z, [1000, 1000, 1001]*um)

    # Specify varying diameters
    cylinder = Cylinder(n, diameter=[10, 5, 2.5]*um,
                        x=[10, 11, 11, 11]*um,
                        y=[100, 100, 101, 101]*um,
                        z=[1000, 1000, 1000, 1001]*um)
    assert_equal(cylinder.diameter, [10, 5, 2.5]*um)


@attr('codegen-independent')
def test_attributes_section():
    n = 10
    # No difference to a cylinder
    sec = Section(n, diameter=10*um, length=200*um)
    assert isinstance(sec, Morphology)
    # Single section with 10 compartments
    assert sec.n == n
    assert sec.n_sections == 1
    assert len(sec) == n

    # Compartment attributes
    assert_equal(sec.diameter, np.ones(n)*10*um)
    assert_equal(sec.length, np.ones(n)*20*um)
    assert_equal(sec.distance, np.arange(n)*20*um + 10*um)
    assert_equal(sec.total_distance, 200*um)
    # TODO: r_length
    assert_allclose(sec.area,
                    np.pi*0.5*(sec.start_diameter + sec.end_diameter)*sec.length)
    assert_allclose(sec.volume, 1.0/4.0*np.pi*sec.diameter**2*sec.length)

    # No coordinates were specified
    assert sec.start_x is None
    assert sec.start_y is None
    assert sec.start_z is None
    assert sec.x is None
    assert sec.y is None
    assert sec.z is None
    assert sec.end_x is None
    assert sec.end_y is None
    assert sec.end_z is None


@attr('codegen-independent')
def test_attributes_section_coordinates_single():
    # Specify only the end-point of the section  (no difference to cylinder)
    n = 10
    # Specify only one of the coordinates
    xyz = {'x', 'y', 'z'}
    for coord in xyz:
        kwds = {coord: 200*um}
        sec = Section(n, diameter=10*um, **kwds)
        assert_equal(sec.diameter, np.ones(n)*10*um)
        assert_equal(sec.length, np.ones(n)*20*um)
        assert_equal(sec.distance, np.arange(n)*20*um + 10*um)
        assert_equal(sec.total_distance, 200*um)

        # Coordinates should be specified now, with 0 values for the other
        # coordinates
        for other_coord in xyz - {coord}:
            assert_equal(getattr(sec, 'start_' + other_coord), np.zeros(n)*um)
            assert_equal(getattr(sec, other_coord), np.zeros(n)*um)
            assert_equal(getattr(sec, 'end_' + other_coord), np.zeros(n)*um)

        assert_equal(getattr(sec, 'start_' + coord), np.arange(n)*20*um)
        assert_equal(getattr(sec, coord), np.arange(n)*20*um + 10*um)
        assert_equal(getattr(sec, 'end_' + coord), np.arange(n)*20*um + 20*um)

    # Specify all coordinates
    val = 200.0/np.sqrt(3.0)*um
    sec = Section(n, diameter=10*um, x=val, y=val, z=val)

    assert_equal(sec.diameter, np.ones(n)*10*um)
    assert_allclose(sec.length, np.ones(n)*20*um)
    assert_allclose(sec.distance, np.arange(n)*20*um + 10*um)
    assert_allclose(sec.total_distance, 200*um)

    for coord in ['x', 'y', 'z']:
        assert_allclose(getattr(sec, 'start_' + coord), np.arange(n)*val/n)
        assert_allclose(getattr(sec, coord), np.arange(n)*val/n + 0.5*val/n)
        assert_allclose(getattr(sec, 'end_' + coord), np.arange(n)*val/n + val/n)


@attr('codegen-independent')
def test_attributes_section_coordinates_endpoints():
    # Specify only the end-point of the section
    n = 3
    # Specify only one of the coordinates
    xyz = {'x', 'y', 'z'}
    for coord in xyz:
        kwds = {coord: [5, 15, 30]*um}
        sec = Section(n, diameter=10*um, **kwds)
        assert_equal(sec.diameter, np.ones(n)*10*um)
        assert_equal(sec.length, [5, 10, 15]*um)
        assert_equal(sec.distance, [2.5, 10, 22.5]*um)
        assert_equal(sec.total_distance, 30*um)

        # Coordinates should be specified now, with 0 values for the other
        # coordinates
        for other_coord in xyz - {coord}:
            assert_equal(getattr(sec, 'start_' + other_coord), np.zeros(n)*um)
            assert_equal(getattr(sec, other_coord), np.zeros(n)*um)
            assert_equal(getattr(sec, 'end_' + other_coord), np.zeros(n)*um)

        assert_equal(getattr(sec, 'start_' + coord), [0, 5, 15]*um)
        assert_equal(getattr(sec, coord), [2.5, 10.0, 22.5]*um)
        assert_equal(getattr(sec, 'end_' + coord), [5, 15, 30]*um)

    # Specify all coordinates
    sec = Section(n, diameter=10*um, x=[1, 1, 1]*um, y=[0, 1, 1]*um,
                        z=[0, 0, 1]*um)

    assert_equal(sec.diameter, np.ones(n)*10*um)
    assert_allclose(sec.length, np.ones(n)*um)
    assert_allclose(sec.distance, np.arange(n)*um + .5*um)
    assert_allclose(sec.total_distance, 3*um)

    assert_equal(sec.start_x, [0, 1, 1]*um)
    assert_equal(sec.x, [0.5, 1, 1]*um)
    assert_equal(sec.end_x, [1, 1, 1]*um)
    assert_equal(sec.start_y, [0, 0, 1]*um)
    assert_equal(sec.y, [0, .5, 1]*um)
    assert_equal(sec.end_y, [0, 1, 1]*um)
    assert_equal(sec.start_z, [0, 0, 0]*um)
    assert_equal(sec.z, [0, 0, 0.5]*um)
    assert_equal(sec.end_z, [0, 0, 1]*um)

    # Specify varying diameters
    sec = Section(n, diameter=[20, 10, 5, 2.5]*um, x=[1, 1, 1]*um,
                  y=[0, 1, 1]*um, z=[0, 0, 1]*um)
    assert_allclose(sec.start_diameter, [20, 10, 5]*um)
    assert_allclose(sec.diameter, [15, 7.5, 3.75]*um)  # average diameter
    assert_allclose(sec.end_diameter, [10, 5, 2.5]*um)
    # TODO: Check area and volume


@attr('codegen-independent')
def test_attributes_section_coordinates_allpoints():
    n = 3
    # Specify all coordinates, including the start point of the section
    sec = Section(n, diameter=10*um,
                        x=[10, 11, 11, 11]*um,
                        y=[100, 100, 101, 101]*um,
                        z=[1000, 1000, 1000, 1001]*um)

    assert_equal(sec.diameter, np.ones(n)*10*um)
    assert_allclose(sec.length, np.ones(n)*um)
    assert_allclose(sec.distance, np.arange(n)*um + .5*um)
    assert_allclose(sec.total_distance, 3*um)

    assert_allclose(sec.start_x, [10, 11, 11]*um)
    assert_allclose(sec.x, [10.5, 11, 11]*um)
    assert_allclose(sec.end_x, [11, 11, 11]*um)
    assert_allclose(sec.start_y, [100, 100, 101]*um)
    assert_allclose(sec.y, [100, 100.5, 101]*um)
    assert_allclose(sec.end_y, [100, 101, 101]*um)
    assert_allclose(sec.start_z, [1000, 1000, 1000]*um)
    assert_allclose(sec.z, [1000, 1000, 1000.5]*um)
    assert_allclose(sec.end_z, [1000, 1000, 1001]*um)

    # Specify varying diameters
    sec = Section(n, diameter=[20, 10, 5, 2.5]*um,
                  x=[10, 11, 11, 11]*um,
                  y=[100, 100, 101, 101]*um,
                  z=[1000, 1000, 1000, 1001]*um)
    assert_allclose(sec.start_diameter, [20, 10, 5]*um)
    assert_allclose(sec.diameter, [15, 7.5, 3.75]*um)  # average diameter
    assert_allclose(sec.end_diameter, [10, 5, 2.5]*um)
    # TODO: Check area and volume


def _check_tree_cables(morphology, coordinates=False, use_cylinders=True):
    # number of compartments per section
    assert morphology.n == 10
    assert morphology.L.n == 5
    assert morphology.R.n == 5
    assert morphology.RL.n == 5
    assert morphology.RR.n == 5
    # number of compartments per subtree
    assert len(morphology) == 30
    assert len(morphology.L) == 5
    assert len(morphology.R) == 15
    assert len(morphology.RL) == 5
    assert len(morphology.RR) == 5
    # number of sections per subtree
    assert morphology.n_sections == 5
    assert morphology.L.n_sections == 1
    assert morphology.R.n_sections == 3
    assert morphology.RL.n_sections == 1
    assert morphology.RR.n_sections == 1
    # Check that distances (= distance to root at electrical midpoint)
    # correctly follow the tree structure
    assert_allclose(morphology.distance, np.arange(10) * 10 * um + 5 * um)
    # TODO: for truncated cones the distance is more complicated
    if use_cylinders:
        assert_allclose(morphology.R.distance,
                        100 * um + np.arange(5) * 10 * um + 5 * um)
        assert_allclose(morphology.RL.distance,
                        150 * um + np.arange(5) * 10 * um + 5 * um)
    assert_allclose(morphology.total_distance, 100 * um)
    assert_allclose(morphology.L.total_distance, 200 * um)
    assert_allclose(morphology.R.total_distance, 150 * um)
    assert_allclose(morphology.RL.total_distance, 200 * um)
    assert_allclose(morphology.RR.total_distance, 200 * um)
    # Check that section diameters are correctly inherited from the parent
    # sections
    assert_allclose(morphology.L.start_diameter, [10, 8, 6, 4, 2] * um)
    assert_allclose(morphology.RR.start_diameter, [5, 4, 3, 2, 1] * um)

    if coordinates:
        # Coordinates should be absolute
        # section: cable
        assert_allclose(morphology.start_x, np.arange(10) * 10 * um)
        assert_allclose(morphology.x, np.arange(10) * 10 * um + 5 * um)
        assert_allclose(morphology.end_x, np.arange(10) * 10 * um + 10 * um)
        assert_allclose(morphology.y, np.zeros(10) * um)
        assert_allclose(morphology.z, np.zeros(10) * um)
        # section: cable.L
        step = 20 / np.sqrt(2) * um
        assert_allclose(morphology.L.start_x, 100 * um + np.arange(5) * step)
        # TODO: x at electrical midpoints
        assert_allclose(morphology.L.end_x, 100 * um + np.arange(5) * step + step)
        assert_allclose(morphology.L.start_y, np.arange(5) * step)
        # TODO: y at electrical midpoints
        assert_allclose(morphology.L.end_y, np.arange(5) * step + step)
        assert_allclose(morphology.L.z, np.zeros(5) * um)
        # section: cable.R
        step = 10 / np.sqrt(2) * um
        assert_allclose(morphology.R.start_x, 100 * um + np.arange(5) * step)
        if use_cylinders:
            assert_allclose(morphology.R.x, 100 * um + np.arange(5) * step + step / 2)
        assert_allclose(morphology.R.end_x, 100 * um + np.arange(5) * step + step)
        assert_allclose(morphology.R.start_y, -np.arange(5) * step)
        if use_cylinders:
            assert_allclose(morphology.R.y, -(np.arange(5) * step + step / 2))
        assert_allclose(morphology.R.end_y, -(np.arange(5) * step + step))
        if use_cylinders:
            assert_allclose(morphology.R.z, np.zeros(5) * um)
        # section: cable.RL
        step = 10 / np.sqrt(2) * um
        assert_allclose(morphology.RL.start_x,
                        100 * um + 50 / np.sqrt(2) * um + np.arange(5) * step)
        if use_cylinders:
            assert_allclose(morphology.RL.x,
                            100 * um + 50 / np.sqrt(2) * um + np.arange(
                                5) * step + step / 2)
        assert_allclose(morphology.RL.end_x,
                        100 * um + 50 / np.sqrt(2) * um + np.arange(
                            5) * step + step)
        assert_allclose(morphology.RL.start_y, -np.ones(5) * 50 / np.sqrt(2) * um)
        if use_cylinders:
            assert_allclose(morphology.RL.y, -np.ones(5) * 50 / np.sqrt(2) * um)
        assert_allclose(morphology.RL.end_y, -np.ones(5) * 50 / np.sqrt(2) * um)
        assert_allclose(morphology.RL.start_z, np.arange(5) * step)
        if use_cylinders:
            assert_allclose(morphology.RL.z, np.arange(5) * step + step / 2)
        assert_allclose(morphology.RL.end_z, np.arange(5) * step + step)
        # section: cable.RR
        step = 10 / np.sqrt(2) * um
        assert_allclose(morphology.RR.start_x,
                        100 * um + 50 / np.sqrt(2) * um + np.arange(5) * step)
        # TODO: x at electrical midpoints
        assert_allclose(morphology.RR.end_x,
                        100 * um + 50 / np.sqrt(2) * um + np.arange(
                            5) * step + step)
        assert_allclose(morphology.RR.start_y, -np.ones(5) * 50 / np.sqrt(2) * um)
        # TODO: y at electrical midpoints
        assert_allclose(morphology.RR.end_y, -np.ones(5) * 50 / np.sqrt(2) * um)
        assert_allclose(morphology.RR.start_z, -np.arange(5) * step)
        # TODO: z at electrical midpoints
        assert_allclose(morphology.RR.end_z, -(np.arange(5) * step + step))


@attr('codegen-independent')
def test_tree_cables_schematic():
    cable = Cylinder(10, diameter=10*um, length=100*um)
    cable.L = Section(5, diameter=[8, 6, 4, 2, 0]*um, length=100*um)  # tapering truncated cones
    cable.R = Cylinder(5, diameter=5*um, length=50*um)
    cable.RL = Cylinder(5, diameter=2.5*um, length=50*um)
    cable.RR = Section(5, diameter=[4, 3, 2, 1, 0]*um, length=50*um)

    _check_tree_cables(cable)

@attr('codegen-independent')
def test_tree_cables_rel_coordinates():
    # The lengths of the sections should be identical to the previous test
    cable = Cylinder(10, x=100*um, diameter=10*um)
    cable.L = Section(5, diameter=[8, 6, 4, 2, 0]*um,
                      x=100/np.sqrt(2)*um, y=100/np.sqrt(2)*um)
    cable.R = Cylinder(5, diameter=5*um, x=50/np.sqrt(2)*um,
                       y=-50/np.sqrt(2)*um)
    cable.RL = Cylinder(5, diameter=2.5*um,
                        x=50/np.sqrt(2)*um,
                        z=50/np.sqrt(2)*um)
    cable.RR = Section(5, diameter=[4, 3, 2, 1, 0]*um,
                       x=50/np.sqrt(2)*um, z=-50/np.sqrt(2)*um)

    _check_tree_cables(cable, coordinates=True)


@attr('codegen-independent')
def test_tree_cables_abs_coordinates():
    # The coordinates should be identical to the previous test
    cable = Cylinder(10, x=np.arange(11)*10*um, diameter=10*um)
    cable.L = Section(5, diameter=[10, 8, 6, 4, 2, 0]*um,
                      x=100*um+np.arange(6)*20/np.sqrt(2)*um,
                      y=np.arange(6)*20/np.sqrt(2)*um)
    cable.R = Cylinder(5, diameter=5*um, x=100*um+np.arange(6)*10/np.sqrt(2)*um,
                       y=-np.arange(6)*10/np.sqrt(2)*um)
    cable.RL = Cylinder(5, diameter=2.5*um,
                        x=100*um+50/np.sqrt(2)*um + np.arange(6)*10/np.sqrt(2)*um,
                        y=-np.ones(6)*50/np.sqrt(2)*um,
                        z=np.arange(6)*10/np.sqrt(2)*um)
    cable.RR = Section(5, diameter=[5, 4, 3, 2, 1, 0]*um,
                       x=100*um+50/np.sqrt(2)*um + np.arange(6)*10/np.sqrt(2)*um,
                       y=-np.ones(6)*50/np.sqrt(2)*um,
                       z=-np.arange(6)*10/np.sqrt(2)*um)

    _check_tree_cables(cable, coordinates=True)


@attr('codegen-independent')
def test_tree_cables_from_points():
    # The coordinates should be identical to the previous test
    points = [ # cable
              (1,  None, 0,                  0,                0,              10, -1),
              (2,  None, 10,                 0,                0,              10,  1),
              (3,  None, 20,                 0,                0,              10,  2),
              (4,  None, 30,                 0,                0,              10,  3),
              (5,  None, 40,                 0,                0,              10,  4),
              (6,  None, 50,                 0,                0,              10,  5),
              (7,  None, 60,                 0,                0,              10,  6),
              (8,  None, 70,                 0,                0,              10,  7),
              (9,  None, 80,                 0,                0,              10,  8),
              (10, None, 90,                 0,                0,              10,  9),
              (11, None, 100,                0,                0,              10,  10),
              # cable.L
              (12, 'L' , 100+20/np.sqrt(2),  20/np.sqrt(2),    0,              8 ,  11),
              (13, 'L' , 100+40/np.sqrt(2),  40/np.sqrt(2),    0,              6 ,  12),
              (14, 'L' , 100+60/np.sqrt(2),  60/np.sqrt(2),    0,              4 ,  13),
              (15, 'L' , 100+80/np.sqrt(2),  80/np.sqrt(2),    0,              2 ,  14),
              (16, 'L' , 100+100/np.sqrt(2), 100/np.sqrt(2),   0,              0 ,  15),
              # cable.R
              (17, 'R' , 100+10/np.sqrt(2),  -10/np.sqrt(2),   0,              5 ,  11),
              (18, 'R' , 100+20/np.sqrt(2),  -20/np.sqrt(2),   0,              5 ,  17),
              (19, 'R' , 100+30/np.sqrt(2),  -30/np.sqrt(2),   0,              5 ,  18),
              (20, 'R' , 100+40/np.sqrt(2),  -40/np.sqrt(2),   0,              5 ,  19),
              (21, 'R' , 100+50/np.sqrt(2),  -50/np.sqrt(2),   0,              5 ,  20),
              # cable.RL
              (22, 'L' , 100+60/np.sqrt(2),  -50/np.sqrt(2),   10/np.sqrt(2),  2.5, 21),
              (23, 'L' , 100+70/np.sqrt(2),  -50/np.sqrt(2),   20/np.sqrt(2),  2.5, 22),
              (24, 'L' , 100+80/np.sqrt(2),  -50/np.sqrt(2),   30/np.sqrt(2),  2.5, 23),
              (25, 'L' , 100+90/np.sqrt(2),  -50/np.sqrt(2),   40/np.sqrt(2),  2.5, 24),
              (26, 'L' , 100+100/np.sqrt(2),  -50/np.sqrt(2),  50/np.sqrt(2),  2.5, 25),
              # cable.RR
              (27, 'R' , 100+60/np.sqrt(2),  -50/np.sqrt(2),   -10/np.sqrt(2), 4,   21),
              (28, 'R' , 100+70/np.sqrt(2),  -50/np.sqrt(2),   -20/np.sqrt(2), 3,   27),
              (29, 'R' , 100+80/np.sqrt(2),  -50/np.sqrt(2),   -30/np.sqrt(2), 2,   28),
              (30, 'R' , 100+90/np.sqrt(2),  -50/np.sqrt(2),   -40/np.sqrt(2), 1,   29),
              (31, 'R' , 100+100/np.sqrt(2),  -50/np.sqrt(2),  -50/np.sqrt(2), 0,   30),
              ]
    cable = Morphology.from_points(points)
    _check_tree_cables(cable, coordinates=True, use_cylinders=False)


def _check_tree_soma(morphology, coordinates=False, use_cylinders=True):

    # number of compartments per section
    assert morphology.n == 1
    assert morphology.L.n == 5
    assert morphology.R.n == 5

    # number of compartments per subtree
    assert len(morphology) == 11
    assert len(morphology.L) == 5
    assert len(morphology.R) == 5

    # number of sections per subtree
    assert morphology.n_sections == 3
    assert morphology.L.n_sections == 1
    assert morphology.R.n_sections == 1

    # Check that distances (= distance to root at electrical midpoint)
    # correctly follow the tree structure
    # Note that the soma does add nothing to the distance
    assert_equal(morphology.distance, 0 * um)
    # TODO: for truncated cones the distance is more complicated
    assert_allclose(morphology.R.distance, np.arange(5)*10*um + 5*um)

    assert_allclose(morphology.total_distance, 0 * um)
    assert_allclose(morphology.L.total_distance, 100 * um)
    assert_allclose(morphology.R.total_distance, 50 * um)

    # Check section diameters
    assert_allclose(morphology.diameter, 30*um)
    assert_allclose(morphology.L.start_diameter, [8, 8, 6, 4, 2]*um)
    assert_allclose(morphology.L.end_diameter,   [8, 6, 4, 2, 0]*um)
    assert_allclose(morphology.R.start_diameter, np.ones(5) * 5*um)
    assert_allclose(morphology.R.diameter, np.ones(5) * 5*um)
    assert_allclose(morphology.R.end_diameter, np.ones(5) * 5*um)

    if coordinates:
        # Coordinates should be absolute
        # section: soma
        assert_allclose(morphology.start_x, 100*um)
        assert_allclose(morphology.x, 100*um)
        assert_allclose(morphology.end_x, 100*um)
        assert_allclose(morphology.y, 0*um)
        assert_allclose(morphology.z, 0*um)
        # section: cable.L
        step = 20 / np.sqrt(2) * um
        assert_allclose(morphology.L.start_x, 100 * um + np.arange(5) * step)
        # TODO: x at electrical midpoints
        assert_allclose(morphology.L.end_x, 100 * um + np.arange(5) * step + step)
        assert_allclose(morphology.L.start_y, np.arange(5) * step)
        # TODO: y at electrical midpoints
        assert_allclose(morphology.L.end_y, np.arange(5) * step + step)
        assert_allclose(morphology.L.z, np.zeros(5) * um)
        # section: cable.R
        step = 10 / np.sqrt(2) * um
        assert_allclose(morphology.R.start_x, 100 * um + np.arange(5) * step)
        if use_cylinders:
            assert_allclose(morphology.R.x, 100 * um + np.arange(5) * step + step / 2)
        assert_allclose(morphology.R.end_x, 100 * um + np.arange(5) * step + step)
        assert_allclose(morphology.R.start_y, -np.arange(5) * step)
        if use_cylinders:
            assert_allclose(morphology.R.y, -(np.arange(5) * step + step / 2))
        assert_allclose(morphology.R.end_y, -(np.arange(5) * step + step))
        if use_cylinders:
            assert_allclose(morphology.R.z, np.zeros(5) * um)


@attr('codegen-independent')
def test_tree_soma_schematic():
    soma = Soma(diameter=30*um)
    soma.L = Section(5, diameter=[8, 8, 6, 4, 2, 0]*um, length=100*um)  # tapering truncated cones
    soma.R = Cylinder(5, diameter=5*um, length=50*um)

    _check_tree_soma(soma)


@attr('codegen-independent')
def test_tree_soma_rel_coordinates():
    soma = Soma(diameter=30*um, x=100*um)
    soma.L = Section(5, diameter=[8, 8, 6, 4, 2, 0]*um,
                     x=100/np.sqrt(2)*um, y=100/np.sqrt(2)*um)  # tapering truncated cones
    soma.R = Cylinder(5, diameter=5*um,
                      x=50/np.sqrt(2)*um, y=-50/np.sqrt(2)*um)

    _check_tree_soma(soma, coordinates=True)


@attr('codegen-independent')
def test_tree_soma_abs_coordinates():
    soma = Soma(diameter=30*um, x=100*um)
    soma.L = Section(5, diameter=[8, 8, 6, 4, 2, 0]*um,
                     x=100*um+np.arange(6)*20/np.sqrt(2)*um,
                     y=np.arange(6)*20/np.sqrt(2)*um)
    soma.R = Cylinder(5, diameter=5*um,
                      x=100*um+np.arange(6)*10/np.sqrt(2)*um,
                      y=-np.arange(6)*10/np.sqrt(2)*um)

    _check_tree_soma(soma, coordinates=True)


@attr('codegen-independent')
def test_tree_soma_from_points():
    # The coordinates should be identical to the previous test
    points = [ # soma
              (1,  'soma', 100,                0,                0,              30, -1),
              # soma.L
              (2,  'L'   , 100+20/np.sqrt(2),  20/np.sqrt(2),    0,              8 ,  1),
              (3,  'L'   , 100+40/np.sqrt(2),  40/np.sqrt(2),    0,              6 ,  2),
              (4,  'L'   , 100+60/np.sqrt(2),  60/np.sqrt(2),    0,              4 ,  3),
              (5,  'L'   , 100+80/np.sqrt(2),  80/np.sqrt(2),    0,              2 ,  4),
              (6,  'L'   , 100+100/np.sqrt(2), 100/np.sqrt(2),   0,              0 ,  5),
              # soma.R
              (7,  'R'   , 100+10/np.sqrt(2),  -10/np.sqrt(2),   0,              5 ,  1),
              (8,  'R'   , 100+20/np.sqrt(2),  -20/np.sqrt(2),   0,              5 ,  7),
              (9,  'R'   , 100+30/np.sqrt(2),  -30/np.sqrt(2),   0,              5 ,  8),
              (10, 'R'   , 100+40/np.sqrt(2),  -40/np.sqrt(2),   0,              5 ,  9),
              (11, 'R'   , 100+50/np.sqrt(2),  -50/np.sqrt(2),   0,              5 ,  10),
              ]
    cable = Morphology.from_points(points)
    _check_tree_soma(cable, coordinates=True, use_cylinders=False)


@attr('codegen-independent')
def test_construction_incorrect_arguments():
    ### Morphology
    dummy_self = Soma(10*um)  # To allow testing of Morphology.__init__
    assert_raises(TypeError, lambda: Morphology.__init__(dummy_self, n=1.5))
    assert_raises(ValueError, lambda: Morphology.__init__(dummy_self, n=0))

    ### Soma
    assert_raises(DimensionMismatchError, lambda: Soma(10))
    assert_raises(TypeError, lambda: Soma([10, 20]*um))
    assert_raises(TypeError, lambda: Soma(x=[10, 20]*um))
    assert_raises(TypeError, lambda: Soma(y=[10, 20]*um))
    assert_raises(TypeError, lambda: Soma(z=[10, 20]*um))
    assert_raises(DimensionMismatchError, lambda: Soma(x=10))
    assert_raises(DimensionMismatchError, lambda: Soma(y=10))
    assert_raises(DimensionMismatchError, lambda: Soma(z=10))

    ### Cylinder
    # Diameter can only be single value or n values
    assert_raises(TypeError, lambda: Cylinder(3, diameter=[10, 20]*um),length=100*um)
    assert_raises(TypeError, lambda: Cylinder(3, diameter=[10, 20, 30, 40]*um), length=100*um)
    # Length can only be single value or n values
    assert_raises(TypeError, lambda: Cylinder(3, diameter=10*um, length=[10, 20]*um))
    assert_raises(TypeError, lambda: Cylinder(3, diameter=10*um, length=[10, 20, 30, 40]*um))
    # Coordinates can be single, n, or n+1 values
    assert_raises(TypeError, lambda: Cylinder(3, diameter=10*um, x=[10, 20]*um))
    assert_raises(TypeError, lambda: Cylinder(3, diameter=10*um, x=[10, 20, 30, 40, 50]*um))
    assert_raises(TypeError, lambda: Cylinder(3, diameter=10*um, y=[10, 20]*um))
    assert_raises(TypeError, lambda: Cylinder(3, diameter=10*um, y=[10, 20, 30, 40, 50]*um))
    assert_raises(TypeError, lambda: Cylinder(3, diameter=10*um, z=[10, 20]*um))
    assert_raises(TypeError, lambda: Cylinder(3, diameter=10*um, z=[10, 20, 30, 40, 50]*um))
    # Need either coordinates or lengths
    assert_raises(TypeError, lambda: Cylinder(3, diameter=10*um))
    # But not both
    assert_raises(TypeError, lambda: Cylinder(3, diameter=10*um, length=[10, 20, 30]*um,
                                              x=[10, 20, 30]*um))

    ### Section
    # Diameter can be a single value, n, or n+1 values
    assert_raises(TypeError, lambda: Section(3, diameter=[10, 20]*um, length=100*um))
    assert_raises(TypeError, lambda: Section(3, diameter=[10, 20, 30, 40, 50]*um, length=100*um))
    # Length can only be single value or n values
    assert_raises(TypeError, lambda: Section(3, diameter=10*um, length=[10, 20]*um))
    assert_raises(TypeError, lambda: Section(3, diameter=10*um, length=[10, 20, 30, 40]*um))
    # Coordinates can be single, n, or n+1 values
    assert_raises(TypeError, lambda: Section(3, diameter=10*um, x=[10, 20]*um))
    assert_raises(TypeError, lambda: Section(3, diameter=10*um, x=[10, 20, 30, 40, 50]*um))
    assert_raises(TypeError, lambda: Section(3, diameter=10*um, y=[10, 20]*um))
    assert_raises(TypeError, lambda: Section(3, diameter=10*um, y=[10, 20, 30, 40, 50]*um))
    assert_raises(TypeError, lambda: Section(3, diameter=10*um, z=[10, 20]*um))
    assert_raises(TypeError, lambda: Section(3, diameter=10*um, z=[10, 20, 30, 40, 50]*um))
    # Need either coordinates or lengths
    assert_raises(TypeError, lambda: Section(3, diameter=10*um))
    # But not both
    assert_raises(TypeError, lambda: Section(3, diameter=10*um, length=[10, 20, 30]*um,
                                              x=[10, 20, 30]*um))
    # If coordinates have been given for the start point, then we need its
    # diameter as well
    assert_raises(TypeError, lambda: Section(3, diameter=[10, 20, 30]*um,
                                             y=[0, 10, 20, 30]*um))

@attr('codegen-independent')
def test_subgroup_indices():
    morpho = Soma(diameter=30*um)
    morpho.L = Cylinder(length=10*um, diameter=1*um, n=10)
    morpho.LL = Cylinder(length=5*um, diameter=2*um, n=5)
    morpho.right = Cylinder(length=3*um, diameter=1*um, n=7)

    assert_equal(morpho.LL.indices[:], [11, 12, 13, 14, 15])
    assert_equal(morpho.L.indices[3*um:5*um], [4, 5])
    assert_equal(morpho.L.indices[3*um:5*um],
                 morpho.L[3*um:5*um].indices[:])
    assert_equal(morpho.L.indices[:5*um], [1, 2, 3, 4, 5])
    assert_equal(morpho.L.indices[3*um:], [4, 5, 6, 7, 8, 9, 10])
    assert_equal(morpho.L.indices[3.5*um], 4)
    assert_equal(morpho.L.indices[3*um], 4)
    assert_equal(morpho.L.indices[3.9*um], 4)
    assert_equal(morpho.L.indices[3], 4)
    assert_equal(morpho.L.indices[-1], 10)
    assert_equal(morpho.L.indices[3:5], [4, 5])
    assert_equal(morpho.L.indices[3:], [4, 5, 6, 7, 8, 9, 10])
    assert_equal(morpho.L.indices[:5], [1, 2, 3, 4, 5])

@attr('codegen-independent')
def test_subgroup_attributes():
    morpho = Soma(diameter=30*um)
    morpho.L = Cylinder(length=10*um, diameter=1*um, n=10)
    morpho.LL = Cylinder(x=5*um, diameter=2*um, n=5)
    morpho.right = Cylinder(length=3*um, diameter=1*um, n=7)

    # # Getting a single compartment by index
    assert_allclose(morpho.L[2].area, morpho.L.area[2])
    assert_allclose(morpho.L[2].volume, morpho.L.volume[2])
    assert_allclose(morpho.L[2].length, morpho.L.length[2])
    assert_allclose(morpho.L[2].r_length, morpho.L.r_length[2])
    assert_allclose(morpho.L[2].distance, morpho.L.distance[2])
    assert_allclose(morpho.L[2].diameter, morpho.L.diameter[2])
    assert_allclose(morpho.L[2].electrical_center, morpho.L.electrical_center[2])
    assert morpho.L[2].x is None
    assert morpho.L[2].y is None
    assert morpho.L[2].z is None
    assert morpho.L[2].start_x is None
    assert morpho.L[2].start_y is None
    assert morpho.L[2].start_z is None
    assert morpho.L[2].end_x is None
    assert morpho.L[2].end_y is None
    assert morpho.L[2].end_z is None

    # # Getting a single compartment by position
    assert_allclose(morpho.LL[1.5*um].area, morpho.LL.area[1])
    assert_allclose(morpho.LL[1.5*um].volume, morpho.LL.volume[1])
    assert_allclose(morpho.LL[1.5*um].length, morpho.LL.length[1])
    assert_allclose(morpho.LL[1.5*um].r_length, morpho.LL.r_length[1])
    assert_allclose(morpho.LL[1.5*um].distance, morpho.LL.distance[1])
    assert_allclose(morpho.LL[1.5*um].diameter, morpho.LL.diameter[1])
    assert_allclose(morpho.LL[1.5*um].electrical_center, morpho.LL.electrical_center[1])
    assert_allclose(morpho.LL[1.5*um].x, morpho.LL.x[1])
    assert_allclose(morpho.LL[1.5*um].y, morpho.LL.y[1])
    assert_allclose(morpho.LL[1.5*um].z, morpho.LL.z[1])
    assert_allclose(morpho.LL[1.5*um].start_x, morpho.LL.start_x[1])
    assert_allclose(morpho.LL[1.5*um].start_y, morpho.LL.start_y[1])
    assert_allclose(morpho.LL[1.5*um].start_z, morpho.LL.start_z[1])
    assert_allclose(morpho.LL[1.5*um].end_x, morpho.LL.end_x[1])
    assert_allclose(morpho.LL[1.5*um].end_y, morpho.LL.end_y[1])
    assert_allclose(morpho.LL[1.5*um].end_z, morpho.LL.end_z[1])

    # Getting several compartments by indices
    assert_allclose(morpho.right[3:6].area, morpho.right.area[3:6])
    assert_allclose(morpho.right[3:6].volume, morpho.right.volume[3:6])
    assert_allclose(morpho.right[3:6].length, morpho.right.length[3:6])
    assert_allclose(morpho.right[3:6].r_length, morpho.right.r_length[3:6])
    assert_allclose(morpho.right[3:6].distance, morpho.right.distance[3:6])
    assert_allclose(morpho.right[3:6].diameter, morpho.right.diameter[3:6])
    assert_allclose(morpho.right[3:6].electrical_center, morpho.right.electrical_center[3:6])
    assert morpho.right[3:6].x is None
    assert morpho.right[3:6].y is None
    assert morpho.right[3:6].z is None
    assert morpho.right[3:6].start_x is None
    assert morpho.right[3:6].start_y is None
    assert morpho.right[3:6].start_z is None
    assert morpho.right[3:6].end_x is None
    assert morpho.right[3:6].end_y is None
    assert morpho.right[3:6].end_z is None

    # Getting several compartments by position
    assert_allclose(morpho.L[3*um:5*um].distance, [3.5, 4.5]*um)

@attr('codegen-independent')
def test_subgroup_incorrect():
    # Incorrect indexing
    morpho = Soma(diameter=30*um)
    morpho.L = Cylinder(length=10*um, diameter=1*um, n=10)
    morpho.LL = Cylinder(length=5*um, diameter=2*um, n=5)
    morpho.right = Cylinder(length=3*um, diameter=1*um, n=7)

    # Non-existing branch
    assert_raises(AttributeError, lambda: morpho.axon)

    # Incorrect indexing
    #  wrong units or mixing units
    assert_raises(TypeError, lambda: morpho.indices[3*second:5*second])
    assert_raises(TypeError, lambda: morpho.indices[3.4:5.3])
    assert_raises(TypeError, lambda: morpho.indices[3:5*um])
    assert_raises(TypeError, lambda: morpho.indices[3*um:5])
    #   providing a step
    assert_raises(TypeError, lambda: morpho.indices[3*um:5*um:2*um])
    assert_raises(TypeError, lambda: morpho.indices[3:5:2])
    #   incorrect type
    assert_raises(TypeError, lambda: morpho.indices[object()])


if __name__ == '__main__':
    test_attributes_soma()
    test_attributes_soma_coordinates()
    test_attributes_cylinder()
    test_attributes_cylinder_coordinates_single()
    test_attributes_cylinder_coordinates_endpoints()
    test_attributes_cylinder_coordinates_allpoints()
    test_attributes_section()
    test_attributes_section_coordinates_single()
    test_attributes_section_coordinates_endpoints()
    test_attributes_section_coordinates_allpoints()
    test_tree_cables_schematic()
    test_tree_cables_rel_coordinates()
    test_tree_cables_abs_coordinates()
    test_tree_cables_from_points()
    test_tree_soma_schematic()
    test_tree_soma_rel_coordinates()
    test_tree_soma_abs_coordinates()
    test_tree_soma_from_points()
    test_construction_incorrect_arguments()
    test_subgroup_indices()
    test_subgroup_attributes()
    test_subgroup_incorrect()
