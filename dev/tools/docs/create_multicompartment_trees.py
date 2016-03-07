import os

import mayavi.mlab as mayavi
from brian2 import *

def find_max_coordinates(morpho, current_max):
    max_x, max_y, max_z = np.max(np.abs(morpho.coordinates), axis=0)
    new_max = (max([max_x, current_max[0]]),
               max([max_y, current_max[1]]),
               max([max_z, current_max[2]]))
    for child in morpho.children:
        new_max = find_max_coordinates(child, new_max)
    return new_max

DARKRED = (0.5450980392156862, 0.0, 0.0)
DARKBLUE = (0.0, 0.0, 0.5450980392156862)

def plot_morphology3D(morpho, color_switch=False, show_compartments=False):
    if isinstance(morpho, Soma):
        mayavi.points3d(morpho.x/um, morpho.y/um, morpho.z/um, morpho.diameter/um,
                        color=DARKRED, scale_factor=1.0, resolution=16)
    else:
        coords = morpho.coordinates
        if color_switch:
            color = DARKBLUE
        else:
            color = DARKRED
        mayavi.plot3d(coords[:, 0]/um, coords[:, 1]/um, coords[:, 2]/um, color=color,
                      tube_radius=1)
        # dots at the center of the compartments
        if show_compartments:
            mayavi.points3d(coords[:, 0]/um, coords[:, 1]/um, coords[:, 2]/um,
                            np.ones(coords.shape[0]), color=color, scale_factor=1,
                            transparent=0.25)

    for child in morpho.children:
        plot_morphology3D(child, color_switch=not color_switch)

def plot_morphology2D(morpho, axis, color_switch=False):
    if isinstance(morpho, Soma):
        ax.plot(morpho.x/um, morpho.y/um, 'o', color='darkred',
                ms=morpho.diameter/um, mec='none')
    else:
        coords = morpho.coordinates
        if color_switch:
            color = 'darkblue'
        else:
            color = 'darkred'
        ax.plot(coords[:, 0]/um, coords[:, 1]/um, color='black',
                lw=2)
        # dots at the center of the compartments
        ax.plot(morpho.x/um, morpho.y/um, 'o', color=color,
                mec='none', alpha=0.75)

    for child in morpho.children:
        plot_morphology2D(child, axis=axis, color_switch=not color_switch)

if __name__ == '__main__':
    dirname = os.path.dirname(__file__)
    PATH = os.path.join(dirname, '..', '..', '..', 'docs_sphinx', 'user', 'images')

    # Construct binary tree according to Rall's formula
    diameter = 32*um
    length = 80*um
    morpho = Soma(diameter=diameter)
    endpoints = {morpho}
    for depth in xrange(1, 10):
        diameter /= 2.**(1./3.)
        length /= 2.**(2./3.)
        new_endpoints = set()
        for endpoint in endpoints:
            new_L = Cylinder(n=max([1, int(length/(5*um))]), diameter=diameter,
                             length=length)
            new_R = Cylinder(n=max([1, int(length/(5*um))]), diameter=diameter,
                             length=length)
            new_endpoints.add(new_L)
            new_endpoints.add(new_R)
            endpoint.L = new_L
            endpoint.R = new_R
        endpoints = new_endpoints
    print 'total number of sections and compartments', morpho.n_sections, len(morpho)
    morpho_with_coords = morpho.generate_coordinates()

    ax = plt.subplot(111)
    plot_morphology2D(morpho_with_coords, ax)
    plt.axis('equal')
    plt.xlabel('x ($\mu$ m)')
    plt.ylabel('y ($\mu$ m)')

    plt.savefig(os.path.join(PATH, 'morphology_deterministic_coords.png'))

    print 'be careful, this plotting takes a long time'
    for title, noise_sec, noise_comp in [('section', 25, 0),
                                         ('section_compartment', 25, 15)]:

        for idx in xrange(3):
            fig = mayavi.figure(bgcolor=(0.95, 0.95, 0.95))
            print idx
            morpho_with_coords = morpho.generate_coordinates(section_randomness=noise_sec,
                                                 compartment_randomness=noise_comp)
            plot_morphology3D(morpho_with_coords)
            cam = fig.scene.camera
            cam.zoom(1.1)
            mayavi.draw()
            mayavi.savefig(os.path.join(PATH, 'morphology_random_%s_%d.png' % (title, idx+1)))
            mayavi.close()
