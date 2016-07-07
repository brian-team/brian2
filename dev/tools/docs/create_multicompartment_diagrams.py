import os

from brian2 import *

HEADER = '''<?xml version="1.0" encoding="iso-8859-1"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 20001102//EN"
 "http://www.w3.org/TR/2000/CR-SVG-20001102/DTD/svg-20001102.dtd">

<svg height="3em" viewBox="{minx} {miny} {width} {height}"
    xmlns="http://www.w3.org/2000/svg"
    xmlns:xlink="http://www.w3.org/1999/xlink/">
'''

# For now, we do not bother showing the actual size of the Soma
SPHERE = '''
    <defs>
      <radialGradient id="blueSphere"
                      cx="0" cy="0" r="100%" fx="-50%" fy="-50%">
        <stop offset="0%" stop-color="white" />
        <stop offset="75%" stop-color="blue" />
        <stop offset="100%" stop-color="darkblue" />
      </radialGradient>
    </defs>
    <circle fill="url(#blueSphere)" cx="0" cy="0" r="5" />
'''

CYLINDER = '''
<g transform="translate({startx},0)">
   <ellipse cx="0"  cy="{center}"
            rx="0.5" ry="{radius}"
            stroke="blue" stroke-width="1"
            fill="blue"/>
   <rect x="0" y="{starty}" width="{length}" height="{diameter}" fill="blue"
    stroke="blue" stroke-width="1"/>
   <ellipse cx="{length}" cy="{center}"
            rx="0.5"  ry="{radius}"
            stroke="darkblue" fill="darkblue" stroke-width="1"/>
</g>
'''

FRUSTRUM = '''
<g transform="translate({startx},0)">
   <ellipse cx="0"  cy="{center}"
            rx="0.5" ry="{radius1}"
            stroke="blue" stroke-width="1"
            fill="blue"/>
   <polygon points="0 {starty1}, 0 {starty2}, {length} {endy2}, {length} {endy1}"
    fill="blue" stroke="blue" stroke-width="1"/>
   <ellipse cx="{length}" cy="{center}"
            rx="0.5"  ry="{radius2}"
            stroke="darkblue" fill="darkblue" stroke-width="1"/>
</g>
'''

PARENT_CYLINDER = '''
<defs>
<linearGradient id="gradient_to_transparent_grey" x1="0" y1="0" x2="100%" y2="0">
    <stop offset="0" stop-color="white" stop-opacity="0"/>
    <stop offset="0.5" stop-color="#222222"  stop-opacity="1"/>
    <stop offset="1" stop-color="#222222"  stop-opacity="1"/>
</linearGradient>
</defs>
<g transform="translate(-{length},0)">
   <rect x="-3" y="{starty}" width="{length}" height="{diameter}" fill="url(#gradient_to_transparent_grey)"
    stroke="url(#gradient_to_transparent_grey)" stroke-width="1"/>
   <g transform="translate(-3,0)">
   <ellipse cx="{length}" cy="{center}"
            rx="0.5"  ry="{radius}"
            stroke="black" fill="black" stroke-width="1"/>
   </g>
</g>
'''

CONNECTION = '''
<line x1="{start}" x2="{end}" y1="{center}" y2="{center}" stroke="black" stroke-width="1"/>
'''
FOOTER = '''
</svg>
'''

def to_svg(morphology):
    if isinstance(morphology, Soma):
        return HEADER.format(minx='-5', miny='-5', width='10', height='10') + SPHERE.format(radius=morphology.diameter[0]/2/um) + FOOTER
    if isinstance(morphology, Cylinder):
        summed_length = 0*um
        elements = []
        center = max([7.5*um, max(morphology.end_diameter)/2])
        minx = -2.5
        if morphology.parent is not None:
            elements.append(PARENT_CYLINDER.format(starty=(center-morphology.parent.end_diameter[-1]/2)/um,
                                                   length=morphology.parent.length[-1]/um,
                                                   center=center/um,
                                                   diameter=morphology.parent.end_diameter[-1]/um,
                                                   radius=morphology.parent.end_diameter[-1]/2/um))
            minx -= morphology.parent.length[-1]/um
        for idx, (diameter, length) in enumerate(zip(morphology.diameter, morphology.length)):
            elements.append(CONNECTION.format(start=(summed_length-3*um)/um,
                                              end=summed_length/um,
                                              center=center/um))
            elements.append(CYLINDER.format(startx=summed_length/um,
                                            starty=(center-diameter/2)/um,
                                            radius=diameter/2/um,
                                            diameter=diameter/um,
                                            length=length/um,
                                            center=center/um))
            summed_length += length + 3*um
        return HEADER.format(minx=minx, miny=-1, width=summed_length/um+0.5-minx, height=center*2/um+2) + ('\n'.join(elements)) + FOOTER
    elif isinstance(morphology, Section):
        summed_length = 0*um
        elements = []
        center = max([7.5*um, max(morphology.end_diameter)/2])
        minx = -5
        if morphology.parent is not None:
                elements.append(PARENT_CYLINDER.format(starty=(center-morphology.parent.end_diameter[-1]/2)/um,
                                                       length=morphology.parent.length[-1]/um,
                                                       center=center/um,
                                                       diameter=morphology.parent.end_diameter[-1]/um,
                                                       radius=morphology.parent.end_diameter[-1]/2/um))
                minx -= morphology.parent.length[-1]/um
        for idx, (start_diameter, end_diameter, length) in enumerate(zip(morphology.start_diameter, morphology.end_diameter, morphology.length)):
            elements.append(CONNECTION.format(start=(summed_length-3*um)/um,
                                              end=summed_length/um,
                                              center=center/um))
            elements.append(FRUSTRUM.format(startx=summed_length/um,
                                            starty1=(center-start_diameter/2)/um,
                                            starty2=(center+start_diameter/2)/um,
                                            endy1=(center-end_diameter/2)/um,
                                            endy2=(center+end_diameter/2)/um,
                                            radius1=start_diameter/2/um,
                                            radius2=end_diameter/2/um,
                                            length=length/um,
                                            center=center/um))
            summed_length += length + 3*um
        return HEADER.format(minx=minx, miny=-1, width=summed_length/um+1.5-minx,
                             height=center*2/um+2) + ('\n'.join(elements)) + FOOTER
    else:
        raise NotImplementedError()
if __name__ == '__main__':
    dirname = os.path.dirname(__file__)
    PATH = os.path.join(dirname, '..', '..', '..', 'docs_sphinx', 'user', 'images')
    root = Cylinder(n=1, diameter=15*um, length=10*um)
    root.cyl = Cylinder(n=5, diameter=10*um, length=50*um)
    root.sec = Section(n=5, diameter=[15, 5, 10, 5, 10, 5]*um,
                       length=[10, 20, 5, 5, 10]*um)
    for filename, morpho in [('soma.svg', Soma(diameter=30*um)),
                             ('cylinder.svg', root.cyl),
                             ('section.svg', root.sec)]:
        with open(os.path.join(PATH, filename), 'w') as f:
            print(filename)
            f.write(to_svg(morpho))
