import os

from brian2 import *


def plot_morphology3D(morpho, axis, color_switch=False, show_compartments=False):
    if isinstance(morpho, Soma):
        ax.plot(morpho.x/um, morpho.y/um, morpho.z/um, 'o', color='darkred',
                ms=morpho.diameter/um, mec='none')
    else:
        coords = morpho.plot_coordinates
        if color_switch:
            color = 'darkblue'
        else:
            color = 'darkred'
        ax.plot(coords[:, 0]/um, coords[:, 1]/um, coords[:, 2]/um, color='black',
                lw=2)
        # dots at the center of the compartments
        if show_compartments:
            ax.plot(morpho.x/um, morpho.y/um, morpho.z/um, 'o', color=color,
                    mec='none', alpha=0.75)

    for child in morpho.children:
        plot_morphology3D(child, axis=axis, color_switch=not color_switch)

def plot_morphology2D(morpho, axis, color_switch=False):
    if isinstance(morpho, Soma):
        ax.plot(morpho.x/um, morpho.y/um, 'o', color='darkred',
                ms=morpho.diameter/um, mec='none')
    else:
        coords = morpho.plot_coordinates
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
    import sys; sys.exit(0)
    print 'be careful, this plotting takes a long time'
    from mpl_toolkits.mplot3d import Axes3D
    for title, noise_sec, noise_comp in [('section', 25, 0),
                                         ('section_compartment', 25, 15)]:
        fig = plt.figure()
        for idx in xrange(3):
            print idx
            morpho_with_coords = morpho.generate_coordinates(section_randomness=noise_sec,
                                                 compartment_randomness=noise_comp)
            ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
            plot_morphology3D(morpho_with_coords, ax)
            ax.set_xlim(-150, 150)
            ax.set_ylim(-150, 150)
            ax.set_zlim(-150, 150)
        plt.tight_layout()
        plt.savefig(os.path.join(PATH, 'morphology_random_%s.png' % title))
