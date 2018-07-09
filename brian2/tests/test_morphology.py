from nose.plugins.attrib import attr
from numpy.testing.utils import assert_equal, assert_raises
import tempfile
import os

from brian2.spatialneuron import *
from brian2.units import um, cm, second, DimensionMismatchError
from brian2 import numpy as np
from brian2.tests.utils import assert_allclose


@attr('codegen-independent')
def test_attributes_soma():
    soma = Soma(diameter=10*um)
    assert isinstance(soma, Morphology)
    # Single compartment
    assert soma.n == 1
    assert soma.total_sections == 1
    assert soma.total_compartments == 1
    assert_raises(TypeError, lambda: len(soma))  # ambiguous
    # Compartment attributes
    assert_equal(soma.diameter, [10]*um)
    assert_equal(soma.length, [10]*um)
    assert_equal(soma.distance, [0]*um)
    assert_equal(soma.end_distance, 0 * um)
    assert soma.r_length_1 > 1*cm
    assert soma.r_length_2 > 1*cm
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
    cylinder = Cylinder(n=n, diameter=10*um, length=200*um)
    assert isinstance(cylinder, Morphology)
    # Single section with 10 compartments
    assert cylinder.n == n
    assert cylinder.total_sections == 1
    assert cylinder.total_compartments == n
    assert_raises(TypeError, lambda: len(cylinder))  # ambiguous

    # Compartment attributes
    assert_equal(cylinder.diameter, np.ones(n)*10*um)
    assert_equal(cylinder.length, np.ones(n)*20*um)
    assert_equal(cylinder.distance, np.arange(n)*20*um + 10*um)
    assert_equal(cylinder.end_distance, 200 * um)
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
def test_attributes_cylinder_coordinates():
    # Specify only the end-point of the section
    n = 10
    # Specify only one of the coordinates
    xyz = {'x', 'y', 'z'}
    for coord in xyz:
        kwds = {coord: [0, 200]*um}
        cylinder = Cylinder(n=n, diameter=10*um, **kwds)
        assert_equal(cylinder.diameter, np.ones(n)*10*um)
        assert_equal(cylinder.length, np.ones(n)*20*um)
        assert_equal(cylinder.distance, np.arange(n)*20*um + 10*um)
        assert_equal(cylinder.end_distance, 200 * um)

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
    val = [0, 200.0/np.sqrt(3.0)]*um
    cylinder = Cylinder(n=n, diameter=10*um, x=val, y=val, z=val)

    assert_equal(cylinder.diameter, np.ones(n)*10*um)
    assert_allclose(cylinder.length, np.ones(n)*20*um)
    assert_allclose(cylinder.distance, np.arange(n)*20*um + 10*um)
    assert_allclose(cylinder.end_distance, 200 * um)

    for coord in ['x', 'y', 'z']:
        assert_allclose(getattr(cylinder, 'start_' + coord), np.arange(n)*val[1]/n)
        assert_allclose(getattr(cylinder, coord), np.arange(n)*val[1]/n + 0.5*val[1]/n)
        assert_allclose(getattr(cylinder, 'end_' + coord), np.arange(n)*val[1]/n + val[1]/n)


@attr('codegen-independent')
def test_attributes_section():
    n = 10
    # No difference to a cylinder
    sec = Section(n=n, diameter=np.ones(n+1)*10*um, length=np.ones(n)*20*um)
    cyl = Cylinder(n=1, diameter=10*um, length=0*um)  # dummy cylinder
    cyl.child = sec
    assert isinstance(sec, Morphology)
    # Single section with 10 compartments
    assert sec.n == n
    assert sec.total_sections == 1
    assert sec.total_compartments == n
    assert_raises(TypeError, lambda: len(sec))  # ambiguous

    # Compartment attributes
    assert_allclose(sec.diameter, np.ones(n)*10*um)
    assert_allclose(sec.length, np.ones(n)*20*um)
    assert_allclose(sec.distance, np.arange(n)*20*um + 10*um)
    assert_allclose(sec.end_distance, 200 * um)
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
        kwds = {coord: np.linspace(0*um, 200*um, n+1)}
        sec = Section(n=n, diameter=np.ones(n+1)*10*um, **kwds)
        cyl = Cylinder(n=1, diameter=10*um, length=0*um)  # dummy cylinder
        cyl.child = sec
        assert_equal(sec.diameter, np.ones(n)*10*um)
        assert_equal(sec.length, np.ones(n)*20*um)
        assert_equal(sec.distance, np.arange(n)*20*um + 10*um)
        assert_equal(sec.end_distance, 200 * um)

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
    sec = Section(n=n, diameter=np.ones(n+1)*10*um,
                  x=np.linspace(0*um, val, n+1),
                  y=np.linspace(0*um, val, n+1),
                  z=np.linspace(0*um, val, n+1))
    cyl = Cylinder(n=1, diameter=10*um, length=0*um)
    cyl.child = sec
    assert_equal(sec.diameter, np.ones(n)*10*um)
    assert_allclose(sec.length, np.ones(n)*20*um)
    assert_allclose(sec.distance, np.arange(n)*20*um + 10*um)
    assert_allclose(sec.end_distance, 200 * um)

    for coord in ['x', 'y', 'z']:
        assert_allclose(getattr(sec, 'start_' + coord), np.arange(n)*val/n)
        assert_allclose(getattr(sec, coord), np.arange(n)*val/n + 0.5*val/n)
        assert_allclose(getattr(sec, 'end_' + coord), np.arange(n)*val/n + val/n)


@attr('codegen-independent')
def test_attributes_section_coordinates_all():
    n = 3
    # Specify all coordinates
    sec = Section(n=n, diameter=[10, 10, 10, 10]*um,
                  x=[10, 11, 11, 11]*um,
                  y=[100, 100, 101, 101]*um,
                  z=[1000, 1000, 1000, 1001]*um)

    assert_equal(sec.diameter, np.ones(n)*10*um)
    assert_allclose(sec.length, np.ones(n)*um)
    assert_allclose(sec.distance, np.arange(n)*um + .5*um)
    assert_allclose(sec.end_distance, 3 * um)

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
    sec = Section(n=n, diameter=[20, 10, 5, 2.5]*um,
                  x=[0, 1, 1, 1]*um, y=[0, 0, 1, 1]*um, z=[0, 0, 0, 1]*um)
    assert_allclose(sec.start_diameter, [20, 10, 5]*um)
    # diameter at midpoint
    assert_allclose(sec.diameter, 0.5*(sec.start_diameter + sec.end_diameter))
    assert_allclose(sec.end_diameter, [10, 5, 2.5]*um)
    # TODO: Check area and volume


def _check_tree_cables(morphology, coordinates=False):
    # number of compartments per section
    assert morphology.n == 10
    assert morphology['1'].n == 5
    assert morphology['2'].n == 5
    assert morphology['21'].n == 5
    assert morphology['22'].n == 5
    # number of compartments per subtree
    assert morphology.total_compartments == 30
    assert morphology['1'].total_compartments == 5
    assert morphology['2'].total_compartments == 15
    assert morphology['21'].total_compartments == 5
    assert morphology['22'].total_compartments == 5
    # number of sections per subtree
    assert morphology.total_sections == 5
    assert morphology['1'].total_sections == 1
    assert morphology['2'].total_sections == 3
    assert morphology['21'].total_sections == 1
    assert morphology['22'].total_sections == 1
    # Check that distances (= distance to root at electrical midpoint)
    # correctly follow the tree structure
    assert_allclose(morphology.distance, np.arange(10) * 10 * um + 5 * um)
    assert_allclose(morphology['2'].distance,
                    100 * um + np.arange(5) * 10 * um + 5 * um)
    assert_allclose(morphology['21'].distance,
                    150 * um + np.arange(5) * 10 * um + 5 * um)
    assert_allclose(morphology.end_distance, 100 * um)
    assert_allclose(morphology['1'].end_distance, 200 * um)
    assert_allclose(morphology['2'].end_distance, 150 * um)
    assert_allclose(morphology['21'].end_distance, 200 * um)
    assert_allclose(morphology['22'].end_distance, 200 * um)
    # Check that section diameters are correctly inherited from the parent
    # sections
    assert_allclose(morphology['1'].start_diameter, [10, 8, 6, 4, 2] * um)
    assert_allclose(morphology['22'].start_diameter, [5, 4, 3, 2, 1] * um)

    if coordinates:
        # Coordinates should be absolute
        # section: cable
        assert_allclose(morphology.start_x, np.arange(10) * 10 * um)
        assert_allclose(morphology.x, np.arange(10) * 10 * um + 5 * um)
        assert_allclose(morphology.end_x, np.arange(10) * 10 * um + 10 * um)
        assert_allclose(morphology.y, np.zeros(10) * um)
        assert_allclose(morphology.z, np.zeros(10) * um)
        # section: cable['1']
        step = 20 / np.sqrt(2) * um
        assert_allclose(morphology['1'].start_x, 100 * um + np.arange(5) * step)
        assert_allclose(morphology['1'].x, 100 * um + np.arange(5) * step + step/2)
        assert_allclose(morphology['1'].end_x, 100 * um + np.arange(5) * step + step)
        assert_allclose(morphology['1'].start_y, np.arange(5) * step)
        assert_allclose(morphology['1'].y, np.arange(5) * step + step/2)
        assert_allclose(morphology['1'].end_y, np.arange(5) * step + step)
        assert_allclose(morphology['1'].z, np.zeros(5) * um)
        # section: cable['2']
        step = 10 / np.sqrt(2) * um
        assert_allclose(morphology['2'].start_x, 100 * um + np.arange(5) * step)
        assert_allclose(morphology['2'].x, 100 * um + np.arange(5) * step + step / 2)
        assert_allclose(morphology['2'].end_x, 100 * um + np.arange(5) * step + step)
        assert_allclose(morphology['2'].start_y, -np.arange(5) * step)
        assert_allclose(morphology['2'].y, -(np.arange(5) * step + step / 2))
        assert_allclose(morphology['2'].end_y, -(np.arange(5) * step + step))
        assert_allclose(morphology['2'].z, np.zeros(5) * um)
        # section: cable ['21']
        step = 10 / np.sqrt(2) * um
        assert_allclose(morphology['21'].start_x,
                        100 * um + 50 / np.sqrt(2) * um + np.arange(5) * step)
        assert_allclose(morphology['21'].x,
                        100 * um + 50 / np.sqrt(2) * um + np.arange(
                            5) * step + step / 2)
        assert_allclose(morphology ['21'].end_x,
                        100 * um + 50 / np.sqrt(2) * um + np.arange(
                            5) * step + step)
        assert_allclose(morphology['21'].start_y, -np.ones(5) * 50 / np.sqrt(2) * um)
        assert_allclose(morphology['21'].y, -np.ones(5) * 50 / np.sqrt(2) * um)
        assert_allclose(morphology['21'].end_y, -np.ones(5) * 50 / np.sqrt(2) * um)
        assert_allclose(morphology['21'].start_z, np.arange(5) * step)
        assert_allclose(morphology['21'].z, np.arange(5) * step + step / 2)
        assert_allclose(morphology['21'].end_z, np.arange(5) * step + step)
        # section: cable['22']
        step = 10 / np.sqrt(2) * um
        assert_allclose(morphology['22'].start_x,
                        100 * um + 50 / np.sqrt(2) * um + np.arange(5) * step)
        assert_allclose(morphology['22'].x,
                        100 * um + 50 / np.sqrt(2) * um + np.arange(5) * step + step/2)
        assert_allclose(morphology['22'].end_x,
                        100 * um + 50 / np.sqrt(2) * um + np.arange(
                            5) * step + step)
        assert_allclose(morphology['22'].start_y, -np.ones(5) * 50 / np.sqrt(2) * um)
        assert_allclose(morphology['22'].y, -np.ones(5) * 50 / np.sqrt(2) * um)
        assert_allclose(morphology['22'].end_y, -np.ones(5) * 50 / np.sqrt(2) * um)
        assert_allclose(morphology['22'].start_z, -np.arange(5) * step)
        assert_allclose(morphology['22'].z, -(np.arange(5) * step + step/2))
        assert_allclose(morphology['22'].end_z, -(np.arange(5) * step + step))


@attr('codegen-independent')
def test_tree_cables_schematic():
    cable = Cylinder(n=10, diameter=10*um, length=100*um)
    cable.L = Section(n=5, diameter=[10, 8, 6, 4, 2, 0]*um, length=np.ones(5)*20*um)  # tapering truncated cones
    cable.R = Cylinder(n=5, diameter=5*um, length=50*um)
    cable.RL = Cylinder(n=5, diameter=2.5*um, length=50*um)
    cable.RR = Section(n=5, diameter=[5, 4, 3, 2, 1, 0]*um, length=np.ones(5)*10*um)

    _check_tree_cables(cable)

@attr('codegen-independent')
def test_tree_cables_coordinates():
    # The lengths of the sections should be identical to the previous test
    cable = Cylinder(n=10, x=[0, 100]*um, diameter=10*um)
    cable.L = Section(n=5, diameter=[10, 8, 6, 4, 2, 0]*um,
                      x=np.linspace(0, 100, 6)/np.sqrt(2)*um,
                      y=np.linspace(0, 100, 6)/np.sqrt(2)*um)
    cable.R = Cylinder(n=5, diameter=5*um, x=[0, 50]*um/np.sqrt(2),
                       y=[0, -50]*um/np.sqrt(2))
    cable.RL = Cylinder(n=5, diameter=2.5*um,
                        x=[0, 50]*um/np.sqrt(2),
                        z=[0, 50]*um/np.sqrt(2))
    cable.RR = Section(n=5, diameter=[5, 4, 3, 2, 1, 0]*um,
                       x=np.linspace(0, 50, 6)*um/np.sqrt(2),
                       z=np.linspace(0, -50, 6)*um/np.sqrt(2))

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
              # cable.L  (using automatic names)
              (12, None, 100+20/np.sqrt(2),  20/np.sqrt(2),    0,              8 ,  11),
              (13, None, 100+40/np.sqrt(2),  40/np.sqrt(2),    0,              6 ,  12),
              (14, None, 100+60/np.sqrt(2),  60/np.sqrt(2),    0,              4 ,  13),
              (15, None, 100+80/np.sqrt(2),  80/np.sqrt(2),    0,              2 ,  14),
              (16, None, 100+100/np.sqrt(2), 100/np.sqrt(2),   0,              0 ,  15),
              # cable.R  (using automatic names)
              (17, None, 100+10/np.sqrt(2),  -10/np.sqrt(2),   0,              5 ,  11),
              (18, None, 100+20/np.sqrt(2),  -20/np.sqrt(2),   0,              5 ,  17),
              (19, None, 100+30/np.sqrt(2),  -30/np.sqrt(2),   0,              5 ,  18),
              (20, None, 100+40/np.sqrt(2),  -40/np.sqrt(2),   0,              5 ,  19),
              (21, None, 100+50/np.sqrt(2),  -50/np.sqrt(2),   0,              5 ,  20),
              # cable.RL (using explicit names)
              (22, 'L' , 100+60/np.sqrt(2),  -50/np.sqrt(2),   10/np.sqrt(2),  2.5, 21),
              (23, 'L' , 100+70/np.sqrt(2),  -50/np.sqrt(2),   20/np.sqrt(2),  2.5, 22),
              (24, 'L' , 100+80/np.sqrt(2),  -50/np.sqrt(2),   30/np.sqrt(2),  2.5, 23),
              (25, 'L' , 100+90/np.sqrt(2),  -50/np.sqrt(2),   40/np.sqrt(2),  2.5, 24),
              (26, 'L' , 100+100/np.sqrt(2),  -50/np.sqrt(2),  50/np.sqrt(2),  2.5, 25),
              # cable.RR (using explicit names)
              (27, 'R' , 100+60/np.sqrt(2),  -50/np.sqrt(2),   -10/np.sqrt(2), 4,   21),
              (28, 'R' , 100+70/np.sqrt(2),  -50/np.sqrt(2),   -20/np.sqrt(2), 3,   27),
              (29, 'R' , 100+80/np.sqrt(2),  -50/np.sqrt(2),   -30/np.sqrt(2), 2,   28),
              (30, 'R' , 100+90/np.sqrt(2),  -50/np.sqrt(2),   -40/np.sqrt(2), 1,   29),
              (31, 'R' , 100+100/np.sqrt(2),  -50/np.sqrt(2),  -50/np.sqrt(2), 0,   30),
              ]
    cable = Morphology.from_points(points)

    # Check that the names are used
    assert cable.L.n == 5
    assert cable.R.n == 5
    assert cable.RL.n == 5
    assert cable.RR.n == 5
    _check_tree_cables(cable, coordinates=True)

def test_tree_cables_from_swc():
    swc_content = '''
# Test file
1   0  0  0  0  5  -1
2   0  10  0  0  5  1
3   0  20  0  0  5  2
4   0  30  0  0  5  3
5   0  40  0  0  5  4
6   0  50  0  0  5  5
7   0  60  0  0  5  6
8   0  70  0  0  5  7
9   0  80  0  0  5  8
10   0  90  0  0  5  9
11   0  100  0  0  5  10
12   2  114.14213562373095  14.142135623730949  0  4  11
13   2  128.2842712474619  28.284271247461898  0  3  12
14   2  142.42640687119285  42.426406871192846  0  2  13
15   2  156.5685424949238  56.568542494923797  0  1  14
16   2  170.71067811865476  70.710678118654741  0  0  15
17   2  107.07106781186548  -7.0710678118654746  0  2.5  11
18   2  114.14213562373095  -14.142135623730949  0  2.5  17
19   2  121.21320343559643  -21.213203435596423  0  2.5  18
20   2  128.2842712474619  -28.284271247461898  0  2.5  19
21   2  135.35533905932738  -35.35533905932737  0  2.5  20
22   2  142.42640687119285  -35.35533905932737  7.0710678118654746  1.25  21
23   2  149.49747468305833  -35.35533905932737  14.142135623730949  1.25  22
24   2  156.5685424949238  -35.35533905932737  21.213203435596423  1.25  23
25   2  163.63961030678928  -35.35533905932737  28.284271247461898  1.25  24
26   2  170.71067811865476  -35.35533905932737  35.35533905932737  1.25  25
27   2  142.42640687119285  -35.35533905932737  -7.0710678118654746  2  21
28   2  149.49747468305833  -35.35533905932737  -14.142135623730949  1.5  27
29   2  156.5685424949238  -35.35533905932737  -21.213203435596423  1  28
30   2  163.63961030678928  -35.35533905932737  -28.284271247461898  0.5  29
31   2  170.71067811865476  -35.35533905932737  -35.35533905932737  0  30
'''
    tmp_filename = tempfile.mktemp('cable_morphology.swc')
    with open(tmp_filename, 'w') as f:
        f.write(swc_content)
    cable = Morphology.from_file(tmp_filename)
    os.remove(tmp_filename)
    _check_tree_cables(cable, coordinates=True)

def _check_tree_soma(morphology, coordinates=False, use_cylinders=True):

    # number of compartments per section
    assert morphology.n == 1
    assert morphology['1'].n == 5
    assert morphology['2'].n == 5

    # number of compartments per subtree
    assert morphology.total_compartments == 11
    assert morphology['1'].total_compartments == 5
    assert morphology['2'].total_compartments == 5

    # number of sections per subtree
    assert morphology.total_sections == 3
    assert morphology['1'].total_sections == 1
    assert morphology['2'].total_sections == 1

    assert_allclose(morphology.diameter, [30]*um)

    # Check that distances (= distance to root at midpoint)
    # correctly follow the tree structure
    # Note that the soma does add nothing to the distance
    assert_equal(morphology.distance, 0 * um)
    assert_allclose(morphology['1'].distance, np.arange(5)*20*um + 10*um)
    assert_allclose(morphology['2'].distance, np.arange(5)*10*um + 5*um)
    assert_allclose(morphology.end_distance, 0 * um)
    assert_allclose(morphology['1'].end_distance, 100 * um)
    assert_allclose(morphology['2'].end_distance, 50 * um)

    assert_allclose(morphology.diameter, 30*um)
    assert_allclose(morphology['1'].start_diameter, [8, 8, 6, 4, 2]*um)
    assert_allclose(morphology['1'].diameter, [8, 7, 5, 3, 1]*um)
    assert_allclose(morphology['1'].end_diameter,   [8, 6, 4, 2, 0]*um)
    assert_allclose(morphology['2'].start_diameter, np.ones(5) * 5*um)
    assert_allclose(morphology['2'].diameter, np.ones(5) * 5*um)
    assert_allclose(morphology['2'].end_diameter, np.ones(5) * 5*um)

    if coordinates:
        # Coordinates should be absolute
        # section: soma
        assert_allclose(morphology.start_x, 100*um)
        assert_allclose(morphology.x, 100*um)
        assert_allclose(morphology.end_x, 100*um)
        assert_allclose(morphology.y, 0*um)
        assert_allclose(morphology.z, 0*um)
        # section: cable['1']
        step = 20 / np.sqrt(2) * um
        assert_allclose(morphology['1'].start_x, 100 * um + np.arange(5) * step)
        assert_allclose(morphology['1'].x, 100 * um + np.arange(5) * step + step/2)
        assert_allclose(morphology['1'].end_x, 100 * um + np.arange(5) * step + step)
        assert_allclose(morphology['1'].start_y, np.arange(5) * step)
        assert_allclose(morphology['1'].y, np.arange(5) * step + step/2)
        assert_allclose(morphology['1'].end_y, np.arange(5) * step + step)
        assert_allclose(morphology['1'].z, np.zeros(5) * um)
        # section: cable['2']
        step = 10 / np.sqrt(2) * um
        assert_allclose(morphology['2'].start_x, 100 * um + np.arange(5) * step)
        if use_cylinders:
            assert_allclose(morphology['2'].x, 100 * um + np.arange(5) * step + step / 2)
        assert_allclose(morphology['2'].end_x, 100 * um + np.arange(5) * step + step)
        assert_allclose(morphology['2'].start_y, -np.arange(5) * step)
        if use_cylinders:
            assert_allclose(morphology['2'].y, -(np.arange(5) * step + step / 2))
        assert_allclose(morphology['2'].end_y, -(np.arange(5) * step + step))
        if use_cylinders:
            assert_allclose(morphology['2'].z, np.zeros(5) * um)


@attr('codegen-independent')
def test_tree_soma_schematic():
    soma = Soma(diameter=30*um)
    soma.L = Section(n=5, diameter=[8, 8, 6, 4, 2, 0]*um,
                     length=np.ones(5)*20*um)  # tapering truncated cones
    soma.R = Cylinder(n=5, diameter=5*um, length=50*um)

    _check_tree_soma(soma)


@attr('codegen-independent')
def test_tree_soma_coordinates():
    soma = Soma(diameter=30*um, x=100*um)
    soma.L = Section(n=5, diameter=[8, 8, 6, 4, 2, 0]*um,
                     x=np.linspace(0, 100, 6)/np.sqrt(2)*um,
                     y=np.linspace(0, 100, 6)/np.sqrt(2)*um)  # tapering truncated cones
    soma.R = Cylinder(n=5, diameter=5*um,
                      x=[0, 50]*um/np.sqrt(2), y=[0, -50]*um/np.sqrt(2))

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
def test_tree_soma_from_points_3_point_soma():
    # The coordinates should be identical to the previous test
    points = [ # soma
              (1,  'soma', 100,                0,                0,              30, -1),
              (2,  'soma', 100,               15,                0,              30,  1),
              (3,  'soma', 100,              -15,                0,              30,  1),
              # soma.L
              (4,  'L'   , 100+20/np.sqrt(2),  20/np.sqrt(2),    0,              8 ,  1),
              (5,  'L'   , 100+40/np.sqrt(2),  40/np.sqrt(2),    0,              6 ,  4),
              (6,  'L'   , 100+60/np.sqrt(2),  60/np.sqrt(2),    0,              4 ,  5),
              (7,  'L'   , 100+80/np.sqrt(2),  80/np.sqrt(2),    0,              2 ,  6),
              (8,  'L'   , 100+100/np.sqrt(2), 100/np.sqrt(2),   0,              0 ,  7),
              # soma.R
              (9,  'R'   , 100+10/np.sqrt(2),  -10/np.sqrt(2),   0,              5 ,  1),
              (10, 'R'   , 100+20/np.sqrt(2),  -20/np.sqrt(2),   0,              5 ,  9),
              (11, 'R'   , 100+30/np.sqrt(2),  -30/np.sqrt(2),   0,              5 ,  10),
              (12, 'R'   , 100+40/np.sqrt(2),  -40/np.sqrt(2),   0,              5 ,  11),
              (13, 'R'   , 100+50/np.sqrt(2),  -50/np.sqrt(2),   0,              5 ,  12),
              ]
    cable = Morphology.from_points(points)
    _check_tree_soma(cable, coordinates=True, use_cylinders=False)
    # The first compartment should be a spherical soma!
    assert isinstance(cable, Soma)


@attr('codegen-independent')
def test_tree_soma_from_points_3_point_soma_incorrect():
    # Inconsistent diameters
    points = [ # soma
              (1,  'soma', 100,                0,                0,              30, -1),
              (2,  'soma', 100,               15,                0,              28,  1),
              (3,  'soma', 100,              -15,                0,              30,  1),
              # soma.L
              (4,  'L'   , 100+20/np.sqrt(2),  20/np.sqrt(2),    0,              8 ,  1),
              (5,  'L'   , 100+40/np.sqrt(2),  40/np.sqrt(2),    0,              6 ,  4),
              (6,  'L'   , 100+60/np.sqrt(2),  60/np.sqrt(2),    0,              4 ,  5),
              (7,  'L'   , 100+80/np.sqrt(2),  80/np.sqrt(2),    0,              2 ,  6),
              (8,  'L'   , 100+100/np.sqrt(2), 100/np.sqrt(2),   0,              0 ,  7)
              ]
    assert_raises(ValueError, lambda: Morphology.from_points(points))

    # Inconsistent coordinates
    points = [  # soma
        (1, 'soma', 100, 0, 0, 30, -1),
        (2, 'soma', 100, 15, 0, 30, 1),
        (3, 'soma', 100, -16, 0, 30, 1),
        # soma.L
        (4, 'L', 100 + 20 / np.sqrt(2), 20 / np.sqrt(2), 0, 8, 1),
        (5, 'L', 100 + 40 / np.sqrt(2), 40 / np.sqrt(2), 0, 6, 4),
        (6, 'L', 100 + 60 / np.sqrt(2), 60 / np.sqrt(2), 0, 4, 5),
        (7, 'L', 100 + 80 / np.sqrt(2), 80 / np.sqrt(2), 0, 2, 6),
        (8, 'L', 100 + 100 / np.sqrt(2), 100 / np.sqrt(2), 0, 0, 7)
    ]
    assert_raises(ValueError, lambda: Morphology.from_points(points))


@attr('codegen-independent')
def test_tree_soma_from_swc():
    swc_content = '''
# Test file
1    1  100  0  0  15  -1
2   2  114.14213562373095  14.142135623730949  0  4  1
3   2  128.2842712474619  28.284271247461898  0  3  2
4   2  142.42640687119285  42.426406871192846  0  2  3
5   2  156.5685424949238  56.568542494923797  0  1  4
6   2  170.71067811865476  70.710678118654741  0  0  5
7   2  107.07106781186548  -7.0710678118654746  0  2.5  1
8   2  114.14213562373095  -14.142135623730949  0  2.5  7
9   2  121.21320343559643  -21.213203435596423  0  2.5  8
10   2  128.2842712474619  -28.284271247461898  0  2.5  9
11   2  135.35533905932738  -35.35533905932737  0  2.5  10
'''
    tmp_filename = tempfile.mktemp('cable_morphology.swc')
    with open(tmp_filename, 'w') as f:
        f.write(swc_content)
    soma = Morphology.from_file(tmp_filename)
    os.remove(tmp_filename)
    _check_tree_soma(soma, coordinates=True, use_cylinders=False)


@attr('codegen-independent')
def test_tree_soma_from_swc_3_point_soma():
    swc_content = '''
# Test file
1    1  100  0  0  15  -1
2    1  100  15  0  15  1
3    1  100  -15  0  15  1
4   2  114.14213562373095  14.142135623730949  0  4  1
5   2  128.2842712474619  28.284271247461898  0  3  4
6   2  142.42640687119285  42.426406871192846  0  2  5
7   2  156.5685424949238  56.568542494923797  0  1  6
8   2  170.71067811865476  70.710678118654741  0  0  7
9   2  107.07106781186548  -7.0710678118654746  0  2.5  1
10   2  114.14213562373095  -14.142135623730949  0  2.5  9
11   2  121.21320343559643  -21.213203435596423  0  2.5  10
12   2  128.2842712474619  -28.284271247461898  0  2.5  11
13   2  135.35533905932738  -35.35533905932737  0  2.5  12
'''
    tmp_filename = tempfile.mktemp('cable_morphology.swc')
    with open(tmp_filename, 'w') as f:
        f.write(swc_content)
    soma = Morphology.from_file(tmp_filename)
    os.remove(tmp_filename)
    _check_tree_soma(soma, coordinates=True, use_cylinders=False)


@attr('codegen-independent')
def test_construction_incorrect_arguments():
    ### Morphology
    dummy_self = Soma(10*um)  # To allow testing of Morphology.__init__
    assert_raises(TypeError, lambda: Morphology.__init__(dummy_self, n=1.5))
    assert_raises(ValueError, lambda: Morphology.__init__(dummy_self, n=0))
    assert_raises(TypeError, lambda: Morphology.__init__(dummy_self,
                                                         'filename.swc'))

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
    # Diameter can only be single value
    assert_raises(TypeError, lambda: Cylinder(n=3, diameter=[10, 20]*um),length=100*um)
    assert_raises(TypeError, lambda: Cylinder(n=3, diameter=[10, 20, 30]*um), length=100*um)
    assert_raises(TypeError, lambda: Cylinder(n=3, diameter=np.ones(3, 2)*um), length=100*um)
    # Length can only be single value
    assert_raises(TypeError, lambda: Cylinder(n=3, diameter=10*um, length=[10, 20]*um))
    assert_raises(TypeError, lambda: Cylinder(n=3, diameter=10*um, length=[10, 20, 30]*um))
    assert_raises(TypeError, lambda: Cylinder(n=3, diameter=10*um, length=np.ones(3, 2)*um))
    # Coordinates have to be two values
    assert_raises(TypeError, lambda: Cylinder(n=3, diameter=10*um, x=[10]*um))
    assert_raises(TypeError, lambda: Cylinder(n=3, diameter=10*um, x=[10, 20, 30]*um))
    assert_raises(TypeError, lambda: Cylinder(n=3, diameter=10*um, y=[10]*um))
    assert_raises(TypeError, lambda: Cylinder(n=3, diameter=10*um, y=[10, 20, 30]*um))
    assert_raises(TypeError, lambda: Cylinder(n=3, diameter=10*um, z=[10]*um))
    assert_raises(TypeError, lambda: Cylinder(n=3, diameter=10*um, z=[10, 20, 30]*um))
    # Need either coordinates or lengths
    assert_raises(TypeError, lambda: Cylinder(n=3, diameter=10*um))
    # But not both
    assert_raises(TypeError, lambda: Cylinder(n=3, diameter=10*um, length=30*um,
                                              x=[0, 30]*um))

    ### Section
    # Diameter have to be n+1 values
    assert_raises(TypeError, lambda: Section(n=3, diameter=10*um, length=np.ones(3)*10*um))
    assert_raises(TypeError, lambda: Section(n=3, diameter=[10, 20, 30]*um, length=np.ones(3)*10*um))
    assert_raises(TypeError, lambda: Section(n=3, diameter=np.ones(4, 2)*um), length=np.ones(3)*10*um)
    # Length have to be n values
    assert_raises(TypeError, lambda: Section(n=3, diameter=np.ones(4)*10*um,
                                             length=10*um))
    assert_raises(TypeError, lambda: Section(n=3, diameter=np.ones(4)*10*um,
                                             length=[10, 20]*um))
    assert_raises(TypeError, lambda: Section(n=3, diameter=np.ones(4)*10*um,
                                             length=np.ones(3, 2)*um))
    # Coordinates have to be n+1 values
    assert_raises(TypeError, lambda: Section(n=3, diameter=np.ones(4)*10*um,
                                             x=10*um))
    assert_raises(TypeError, lambda: Section(n=3, diameter=np.ones(4)*10*um,
                                             x=[10, 20, 30]*um))
    assert_raises(TypeError, lambda: Section(n=3, diameter=np.ones(4)*10*um,
                                             y=10*um))
    assert_raises(TypeError, lambda: Section(n=3, diameter=np.ones(4)*10*um,
                                             y=[10, 20, 30]*um))
    assert_raises(TypeError, lambda: Section(n=3, diameter=np.ones(4)*10*um,
                                             z=10*um))
    assert_raises(TypeError, lambda: Section(n=3, diameter=np.ones(4)*10*um,
                                             z=[10, 20, 30]*um))
    # Need either coordinates or lengths
    assert_raises(TypeError, lambda: Section(n=3, diameter=np.ones(4)*10*um))
    # But not both
    assert_raises(TypeError, lambda: Section(n=3, diameter=np.ones(4)*10*um,
                                             length=[10, 20, 30]*um,
                                             x=[0, 10, 20, 30]*um))


@attr('codegen-independent')
def test_from_points_minimal():
    points = [(1, 'soma', 10, 20, 30,  30,  -1)]
    morph = Morphology.from_points(points)
    assert morph.total_compartments == 1
    assert_allclose(morph.diameter, 30*um)
    assert_allclose(morph.x, 10*um)
    assert_allclose(morph.y, 20*um)
    assert_allclose(morph.z, 30*um)


@attr('codegen-independent')
def test_from_points_incorrect():
    # The coordinates should be identical to the previous test
    points = [
              (1,  None, 0,                  0,                0,              10, -1),
              (2,  None, 10,                 0,                0,              10,  1),
              (2,  None, 20,                 0,                0,              10,  2),
              ]
    points2 = [
              (1,  None, 0,                  0,                0,              10, -1),
              (2,  None, 10,                 0,                0,              10,  1),
              (3,  None, 20,                 0,                0,              10,  3),
              ]
    points3 = [
              (1,  None, 0,                  0,                0,              10, -1),
              (2,  None, 10,                 0,                0,              10,  1),
              (3,  None, 20,                 0,                0,              10,  4),
              ]
    points4 = [
              (1,  0,                  0,                0,              10, -1),
              (2,  10,                 0,                0,              10,  1),
              (3,  20,                 0,                0,              10,  2),
              ]
    assert_raises(ValueError, lambda: Morphology.from_points(points))
    assert_raises(ValueError, lambda: Morphology.from_points(points2))
    assert_raises(ValueError, lambda: Morphology.from_points(points3))
    assert_raises(ValueError, lambda: Morphology.from_points(points4))


@attr('codegen-independent')
def test_subtree_deletion():
    soma = Soma(diameter=30*um)
    first_dendrite = Cylinder(n=5, diameter=5*um, length=50*um)
    second_dendrite = Cylinder(n=5, diameter=5*um, length=50*um)
    second_dendrite.L = Cylinder(n=5, diameter=5*um, length=50*um)
    second_dendrite.R = Cylinder(n=5, diameter=5*um, length=50*um)
    soma.dend1 = first_dendrite
    soma.dend2 = second_dendrite
    soma.dend3 = Cylinder(n=5, diameter=5*um, length=50*um)
    soma.dend3.L = Cylinder(n=5, diameter=5*um, length=50*um)
    soma.dend3.L.L = Cylinder(n=5, diameter=5 * um, length=50 * um)

    assert soma.total_compartments == 36

    del soma.dend1
    assert soma.total_compartments == 31
    assert_raises(AttributeError, lambda: soma.dend1)
    assert_raises(AttributeError, lambda: delattr(soma, 'dend1'))
    assert_raises(AttributeError, lambda: soma.__delitem__('dend1'))
    assert first_dendrite not in soma.children

    del soma['dend2']
    assert soma.total_compartments == 16
    assert_raises(AttributeError, lambda: soma.dend2)
    assert second_dendrite not in soma.children

    del soma.dend3.LL
    assert soma.total_compartments == 11
    assert_raises(AttributeError, lambda: soma.dend3.LL)
    assert_raises(AttributeError, lambda: soma.dend3.L.L)


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
    morpho.LL = Cylinder(x=[0, 5]*um, diameter=2*um, n=5)
    morpho.right = Cylinder(length=3*um, diameter=1*um, n=7)

    # # Getting a single compartment by index
    assert_allclose(morpho.L[2].area, morpho.L.area[2])
    assert_allclose(morpho.L[2].volume, morpho.L.volume[2])
    assert_allclose(morpho.L[2].length, morpho.L.length[2])
    assert_allclose(morpho.L[2].r_length_1, morpho.L.r_length_1[2])
    assert_allclose(morpho.L[2].r_length_2, morpho.L.r_length_2[2])
    assert_allclose(morpho.L[2].distance, morpho.L.distance[2])
    assert_allclose(morpho.L[2].diameter, morpho.L.diameter[2])
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
    assert_allclose(morpho.LL[1.5*um].r_length_1, morpho.LL.r_length_1[1])
    assert_allclose(morpho.LL[1.5*um].r_length_2, morpho.LL.r_length_2[1])
    assert_allclose(morpho.LL[1.5*um].distance, morpho.LL.distance[1])
    assert_allclose(morpho.LL[1.5*um].diameter, morpho.LL.diameter[1])
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
    assert_allclose(morpho.right[3:6].r_length_1, morpho.right.r_length_1[3:6])
    assert_allclose(morpho.right[3:6].r_length_2, morpho.right.r_length_2[3:6])
    assert_allclose(morpho.right[3:6].distance, morpho.right.distance[3:6])
    assert_allclose(morpho.right[3:6].diameter, morpho.right.diameter[3:6])
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
    assert_allclose(morpho.L[3.5*um:4.5*um].distance, [3.5, 4.5]*um)


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
    assert_raises(TypeError, lambda: morpho.L[3*second:5*second])
    assert_raises(TypeError, lambda: morpho.L[3.4:5.3])
    assert_raises(TypeError, lambda: morpho.L[3:5*um])
    assert_raises(TypeError, lambda: morpho.L[3*um:5])
    #   providing a step
    assert_raises(TypeError, lambda: morpho.L[3*um:5*um:2*um])
    assert_raises(TypeError, lambda: morpho.L[3:5:2])
    #   incorrect type
    assert_raises(TypeError, lambda: morpho.L[object()])
    # out of range
    assert_raises(IndexError, lambda: morpho.L[-10*um])
    assert_raises(IndexError, lambda: morpho.L[15*um])
    assert_raises(IndexError, lambda: morpho.L[10])


@attr('codegen-independent')
def test_topology():
    soma = Soma(diameter=30*um)
    soma.L = Section(n=5, diameter=[10, 8, 6, 4, 2, 0]*um,
                     length=np.ones(5)*20*um)  # tapering truncated cones
    soma.R = Cylinder(n=10, diameter=5*um, length=50*um)
    soma.R.left = Cylinder(n=10, diameter=2.5*um, length=50*um)
    soma.R.right = Section(n=5, diameter=[5, 4, 3, 2, 1, 0]*um,
                           length=np.ones(5)*10*um)

    str_topology = str(soma.topology())
    lines = [l for l in str_topology.split('\n') if len(l.strip())]
    assert len(lines) == 5  # one line for each section
    for line, name in zip(lines, ['root', '.L', '.R', '.R.left', 'R.right']):
        assert name in line


@attr('codegen-independent')
def test_copy_section_soma():
    soma = Soma(diameter=30*um)
    soma_copy = soma.copy_section()
    assert soma_copy.diameter[0] == 30*um
    assert soma_copy.x is None
    assert soma_copy.y is None
    assert soma_copy.z is None
    assert soma_copy.type == 'soma'

    soma = Soma(diameter=30*um, x=5*um, z=-10*um)
    soma_copy = soma.copy_section()
    assert soma_copy.diameter[0] == 30*um
    assert_allclose(soma_copy.x[0], 5*um)
    assert_allclose(soma_copy.y[0], 0*um)
    assert_allclose(soma_copy.z[0], -10*um)
    assert soma_copy.type == 'soma'


@attr('codegen-independent')
def test_copy_section_section():
    # No coordinates
    sec = Section(diameter=[10, 5, 4, 3, 2, 1]*um, n=5,
                  length=np.ones(5)*10*um, type='dend')
    sec_copy = sec.copy_section()
    assert_allclose(sec_copy.start_diameter, sec.start_diameter)
    assert_allclose(sec_copy.end_diameter, sec.end_diameter)
    assert_allclose(sec_copy.length, sec.length)
    assert sec_copy.n == sec.n
    assert sec_copy.x is None
    assert sec_copy.y is None
    assert sec_copy.z is None
    assert sec_copy.type == 'dend'

    # With coordinates
    sec = Section(diameter=[10, 5, 4, 3, 2, 1]*um, n=5,
                  x=[0, 1, 2, 3, 4, 5]*um,
                  y=[0, -1, -2, -3, -4, -5]*um)
    sec_copy = sec.copy_section()
    assert_allclose(sec_copy.start_diameter, sec.start_diameter)
    assert_allclose(sec_copy.end_diameter, sec.end_diameter)
    assert_allclose(sec_copy.length, sec.length)
    assert sec_copy.n == sec.n
    assert_allclose(sec_copy.x, sec.x)
    assert_allclose(sec_copy.y, sec.y)
    assert_allclose(sec_copy.z, sec.z)

    assert sec_copy.type is None

@attr('codegen-independent')
def test_copy_section_cylinder():
    # no coordinates
    sec = Section(diameter=[10, 5, 4, 3, 2, 1]*um, n=5,
                  length=np.ones(5)*20*um, type='dend')
    sec_copy = sec.copy_section()
    assert_allclose(sec_copy.end_diameter, sec.end_diameter)
    assert_allclose(sec_copy.length, sec.length)
    assert sec_copy.n == sec.n
    assert sec_copy.x is None
    assert sec_copy.y is None
    assert sec_copy.z is None
    assert sec_copy.type == 'dend'

    # with coordinates
    sec = Section(diameter=[10, 5, 4, 3, 2, 1]*um, n=5,
                  x=[0, 1, 2, 3, 4, 5]*um, y=[0, -1, -2, -3, -4, -5]*um)
    sec_copy = sec.copy_section()
    assert_allclose(sec_copy.end_diameter, sec.end_diameter)
    assert_allclose(sec_copy.length, sec.length)
    assert sec_copy.n == sec.n
    assert_allclose(sec_copy.x, sec.x)
    assert_allclose(sec_copy.y, sec.y)
    assert_allclose(sec_copy.z, sec.z)

    assert sec_copy.type is None


def _check_length_coord_consistency(morph_with_coords):
    if not isinstance(morph_with_coords, Soma):
        vectors = np.diff(morph_with_coords.coordinates, axis=0)
        calculated_length = np.sqrt(np.sum(vectors**2, axis=1))
        assert_allclose(calculated_length, morph_with_coords.length)
    for child in morph_with_coords.children:
        _check_length_coord_consistency(child)


@attr('codegen-independent')
def test_generate_coordinates_deterministic():
    morph = Soma(diameter=30*um)
    morph.L = Section(n=5, diameter=[10, 8, 6, 4, 2, 0]*um,
                      length=np.ones(5)*20*um)  # tapering truncated cones
    morph.R = Cylinder(n=10, diameter=5*um, length=50*um)
    morph.R.left = Cylinder(n=10, diameter=2.5*um, length=50*um)
    morph.R.right = Section(n=5, diameter=[5, 4, 3, 2, 1, 0]*um,
                            length=np.ones(5)*10*um)

    morph_with_coords = morph.generate_coordinates()
    assert morph_with_coords.total_compartments == morph.total_compartments
    assert morph_with_coords.total_sections == morph.total_sections

    for new, old in [(morph_with_coords, morph),
                     (morph_with_coords.L, morph.L),
                     (morph_with_coords.R, morph.R),
                     (morph_with_coords.R.left, morph.R.left),
                     (morph_with_coords.R.right, morph.R.right)]:
        assert new.n == old.n
        assert_allclose(new.length, old.length)
        assert_allclose(new.diameter, old.diameter)
        # The morphology should be in the x/y plane
        assert_equal(new.z, 0*um)

    _check_length_coord_consistency(morph_with_coords)


@attr('codegen-independent')
def test_generate_coordinates_random_sections():
    morph = Soma(diameter=30*um)
    morph.L = Section(n=5, diameter=[10, 8, 6, 4, 2, 0]*um,
                      length=np.ones(5)*20*um)  # tapering truncated cones
    morph.R = Cylinder(n=10, diameter=5*um, length=50*um)
    morph.R.left = Cylinder(n=10, diameter=2.5*um, length=50*um)
    morph.R.right = Section(n=5, diameter=[5, 4, 3, 2, 1, 0]*um,
                            length=np.ones(5)*10*um)

    morph_with_coords = morph.generate_coordinates(section_randomness=25)
    assert morph_with_coords.total_compartments == morph.total_compartments
    assert morph_with_coords.total_sections == morph.total_sections

    for new, old in [(morph_with_coords, morph),
                     (morph_with_coords.L, morph.L),
                     (morph_with_coords.R, morph.R),
                     (morph_with_coords.R.left, morph.R.left),
                     (morph_with_coords.R.right, morph.R.right)]:
        assert new.n == old.n
        assert_allclose(new.length, old.length)
        assert_allclose(new.diameter, old.diameter)

    _check_length_coord_consistency(morph_with_coords)


@attr('codegen-independent')
def test_generate_coordinates_random_compartments():
    morph = Soma(diameter=30*um)
    morph.L = Section(n=5, diameter=[10, 8, 6, 4, 2, 0]*um,
                      length=np.ones(5)*20*um)  # tapering truncated cones
    morph.R = Cylinder(n=10, diameter=5*um, length=50*um)
    morph.R.left = Cylinder(n=10, diameter=2.5*um, length=50*um)
    morph.R.right = Section(n=5, diameter=[5, 4, 3, 2, 1, 0]*um,
                            length=np.ones(5)*10*um)

    morph_with_coords = morph.generate_coordinates(compartment_randomness=15)
    assert morph_with_coords.total_compartments == morph.total_compartments
    assert morph_with_coords.total_sections == morph.total_sections

    for new, old in [(morph_with_coords, morph),
                     (morph_with_coords.L, morph.L),
                     (morph_with_coords.R, morph.R),
                     (morph_with_coords.R.left, morph.R.left),
                     (morph_with_coords.R.right, morph.R.right)]:
        assert new.n == old.n
        assert_allclose(new.length, old.length)
        assert_allclose(new.diameter, old.diameter)

    _check_length_coord_consistency(morph_with_coords)


@attr('codegen-independent')
def test_generate_coordinates_random_all():
    morph = Soma(diameter=30*um)
    morph.L = Section(n=5, diameter=[10, 8, 6, 4, 2, 0]*um,
                      length=np.ones(5)*20*um)  # tapering truncated cones
    morph.R = Cylinder(n=10, diameter=5*um, length=50*um)
    morph.R.left = Cylinder(n=10, diameter=2.5*um, length=50*um)
    morph.R.right = Section(n=5, diameter=[5, 4, 3, 2, 1, 0]*um,
                            length=np.ones(5)*10*um)

    morph_with_coords = morph.generate_coordinates(section_randomness=25,
                                                   compartment_randomness=15)
    assert morph_with_coords.total_compartments == morph.total_compartments
    assert morph_with_coords.total_sections == morph.total_sections

    for new, old in [(morph_with_coords, morph),
                     (morph_with_coords.L, morph.L),
                     (morph_with_coords.R, morph.R),
                     (morph_with_coords.R.left, morph.R.left),
                     (morph_with_coords.R.right, morph.R.right)]:
        assert new.n == old.n
        assert_allclose(new.length, old.length)
        assert_allclose(new.diameter, old.diameter)

    _check_length_coord_consistency(morph_with_coords)


@attr('codegen-independent')
def test_generate_coordinates_no_overwrite():
    morph = Soma(diameter=30*um)
    morph.L = Section(n=5, diameter=[10, 8, 6, 4, 2, 0]*um,
                      length=np.ones(5)*20*um)  # tapering truncated cones
    morph.R = Cylinder(n=10, diameter=5*um, length=50*um)
    morph.R.left = Cylinder(n=10, diameter=2.5*um, length=50*um)
    morph.R.right = Section(n=5, diameter=[5, 4, 3, 2, 1, 0]*um,
                            length=np.ones(5)*10*um)

    morph_with_coords = morph.generate_coordinates(compartment_randomness=15)
    # This should not change anything because the morphology already has coordinates!
    morph_with_coords2 = morph_with_coords.generate_coordinates(section_randomness=25,
                                                                compartment_randomness=15)

    for new, old in [(morph_with_coords2, morph_with_coords),
                     (morph_with_coords2.L, morph_with_coords.L),
                     (morph_with_coords2.R, morph_with_coords.R),
                     (morph_with_coords2.R.left, morph_with_coords.R.left),
                     (morph_with_coords2.R.right, morph_with_coords.R.right)]:
        assert new.n == old.n
        assert_allclose(new.length, old.length)
        assert_allclose(new.diameter, old.diameter)
        assert_allclose(new.x, old.x)
        assert_allclose(new.y, old.y)
        assert_allclose(new.z, old.z)


@attr('codegen-independent')
def test_generate_coordinates_overwrite():
    morph = Soma(diameter=30*um)
    morph.L = Section(n=5, diameter=[10, 8, 6, 4, 2, 0]*um,
                      length=np.ones(5)*20*um)  # tapering truncated cones
    morph.R = Cylinder(n=10, diameter=5*um, length=50*um)
    morph.R.left = Cylinder(n=10, diameter=2.5*um, length=50*um)
    morph.R.right = Section(n=5, diameter=[5, 4, 3, 2, 1, 0]*um,
                            length=np.ones(5)*10*um)

    morph_with_coords = morph.generate_coordinates(compartment_randomness=15)
    # This should change things since we explicitly ask for it
    morph_with_coords2 = morph_with_coords.generate_coordinates(section_randomness=25,
                                                                compartment_randomness=15,
                                                                overwrite_existing=True)

    for new, old in [# ignore the root compartment
                     (morph_with_coords2.L, morph_with_coords.L),
                     (morph_with_coords2.R, morph_with_coords.R),
                     (morph_with_coords2.R.left, morph_with_coords.R.left),
                     (morph_with_coords2.R.right, morph_with_coords.R.right)]:
        assert new.n == old.n
        assert_allclose(new.length, old.length)
        assert_allclose(new.diameter, old.diameter)
        assert all(np.abs(new.x - old.x) > 0)
        assert all(np.abs(new.y - old.y) > 0)
        assert all(np.abs(new.z - old.z) > 0)

    _check_length_coord_consistency(morph_with_coords2)


@attr('codegen-independent')
def test_generate_coordinates_mixed_overwrite():
    morph = Soma(diameter=30*um)
    morph.L = Section(n=5, diameter=[10, 8, 6, 4, 2, 0]*um,
                      length=np.ones(5)*20*um)  # tapering truncated cones
    morph.R = Cylinder(n=10, diameter=5*um, length=50*um)
    morph_with_coords = morph.generate_coordinates(section_randomness=25,
                                                   compartment_randomness=15)
    # The following just returns a copy, as all coordinates are already
    # specified
    morph_copy = morph_with_coords.generate_coordinates()

    # Add new sections that do not yet have coordinates
    morph_with_coords.R.left = Cylinder(n=10, diameter=2.5*um, length=50*um)
    morph_with_coords.R.right = Section(n=5, diameter=[5, 4, 3, 2, 1, 0]*um,
                                        length=np.ones(5)*10*um)

    # This should change things since we explicitly ask for it
    morph_with_coords2 = morph_with_coords.generate_coordinates(section_randomness=25,
                                                                compartment_randomness=15)

    for new, old in [(morph_with_coords2, morph_with_coords),
                     (morph_with_coords2.L, morph_with_coords.L),
                     (morph_with_coords2.R, morph_with_coords.R)]:
        assert new.n == old.n
        assert_allclose(new.length, old.length)
        assert_allclose(new.diameter, old.diameter)
        assert_allclose(new.x, old.x)
        assert_allclose(new.y, old.y)
        assert_allclose(new.z, old.z)

    assert morph_with_coords.R.left.x is None
    assert len(morph_with_coords2.R.left.x) == morph_with_coords2.R.left.n

    _check_length_coord_consistency(morph_with_coords2)


@attr('codegen-independent')
def test_str_repr():
    # A very basic test, make sure that the str/repr functions return
    # something and do not raise an error
    for morph in [Soma(diameter=30*um),
                  Soma(diameter=30*um, x=5*um, y=10*um),
                  Cylinder(n=5, diameter=10*um, length=50*um),
                  Cylinder(n=5, diameter=10*um, x=[0, 50]*um),
                  Section(n=5, diameter=[2.5, 5, 10, 5, 10, 5]*um, length=[10, 20, 5, 5, 10]*um),
                  Section(n=5, diameter=[2.5, 5, 10, 5, 10, 5]*um, x=[0, 10, 30, 35, 40, 50]*um)]:

        assert len(repr(morph)) > 0
        assert len(str(morph)) > 0
    morph = Soma(30*um)
    assert len(repr(morph.children)) > 0
    assert len(str(morph.children)) > 0
    morph.axon = Cylinder(1*um, n=10, length=100*um)
    morph.dend = Cylinder(1*um, n=10, length=50*um)
    assert len(repr(morph.children)) > 0
    assert len(str(morph.children)) > 0


if __name__ == '__main__':
    test_attributes_soma()
    test_attributes_soma_coordinates()
    test_attributes_cylinder()
    test_attributes_cylinder_coordinates()
    test_attributes_section()
    test_attributes_section_coordinates_single()
    test_attributes_section_coordinates_all()
    test_tree_cables_schematic()
    test_tree_cables_coordinates()
    test_tree_cables_from_points()
    test_tree_cables_from_swc()
    test_tree_soma_schematic()
    test_tree_soma_coordinates()
    test_tree_soma_from_points()
    test_tree_soma_from_points_3_point_soma()
    test_tree_soma_from_points_3_point_soma_incorrect()
    test_tree_soma_from_swc()
    test_tree_soma_from_swc_3_point_soma()
    test_construction_incorrect_arguments()
    test_from_points_minimal()
    test_from_points_incorrect()
    test_subtree_deletion()
    test_subgroup_indices()
    test_subgroup_attributes()
    test_subgroup_incorrect()
    test_topology()
    test_copy_section_soma()
    test_copy_section_section()
    test_copy_section_cylinder()
    test_generate_coordinates_deterministic()
    test_generate_coordinates_random_sections()
    test_generate_coordinates_random_compartments()
    test_generate_coordinates_random_all()
    test_generate_coordinates_no_overwrite()
    test_generate_coordinates_overwrite()
    test_generate_coordinates_mixed_overwrite()
    test_str_repr()
