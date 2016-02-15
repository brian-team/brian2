from brian2 import *

HEADER = '''<?xml version="1.0" encoding="iso-8859-1"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 20001102//EN"
 "http://www.w3.org/TR/2000/CR-SVG-20001102/DTD/svg-20001102.dtd">

<svg width="100%" height="100%" viewBox="{minx} {miny} {width} {height}"
    xmlns="http://www.w3.org/2000/svg"
    xmlns:xlink="http://www.w3.org/1999/xlink/"
    baseProfile="tiny" version="1.2">
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

FRUSTRUM_NO_START = '''
<defs>
<linearGradient id="gradient_to_transparent" x1="0%" x2="100%">
    <stop offset="0.5" stop-opacity="1" stop-color="blue"></stop>
    <stop offset="0" stop-opacity="0" stop-color="blue"></stop>
</linearGradient>
</defs>
<g transform="translate({startx},0)">
   <polygon points="0 {starty1}, 0 {starty2}, {length} {endy2}, {length} {endy1}"
    fill="url(#gradient_to_transparent)" stroke="url(#gradient_to_transparent)" stroke-width="1"/>
   <ellipse cx="{length}" cy="{center}"
            rx="0.5"  ry="{radius2}"
            stroke="darkblue" fill="darkblue" stroke-width="1"/>
   <text x="0" y="{center}" fill="darkblue" dominant-baseline="central" alignment-baseline="central" text-anchor="middle"
    style="font-size: 8px">?</text>
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
        center = np.max(morphology.diameter)/2
        for idx, (diameter, length) in enumerate(zip(morphology.diameter, morphology.length)):
            elements.append(CONNECTION.format(start=str((summed_length-3*um)/um),
                                              end=str(summed_length/um),
                                              center=str(center/um)))
            elements.append(CYLINDER.format(startx=str(summed_length/um),
                                            starty=str((center-diameter/2)/um),
                                            radius=str(diameter/2/um),
                                            diameter=str(diameter/um),
                                            length=str(length/um),
                                            center=str(center/um)))
            summed_length += length + 3*um
        return HEADER.format(minx=-2.5, miny=0, width=summed_length/um+0.5, height=center*2/um) + ('\n'.join(elements)) + FOOTER
    elif isinstance(morphology, Section):
        summed_length = 0*um
        elements = []
        center = max(morphology.end_diameter)/2
        for idx, (start_diameter, end_diameter, length) in enumerate(zip(morphology.start_diameter, morphology.end_diameter, morphology.length)):
            if isnan(start_diameter):
                elements.append(FRUSTRUM_NO_START.format(startx=summed_length/um,
                                                         starty1=(center-end_diameter)/um,
                                                         starty2=(center+end_diameter)/um,
                                                         endy1=(center-end_diameter/2)/um,
                                                         endy2=(center+end_diameter/2)/um,
                                                         radius1=end_diameter/2/um,
                                                         radius2=end_diameter/2/um,
                                                         length=length/um,
                                                         center=center/um))
            else:
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
        return HEADER.format(minx=-2.5, miny=0, width=summed_length/um+1.5,
                             height=center*2/um) + ('\n'.join(elements)) + FOOTER
    else:
        raise NotImplementedError()

if __name__ == '__main__':
    # morpho = Cylinder(5, diameter=[5, 10, 5, 10, 5]*um, length=[10, 20, 5, 5, 10]*um)
    morpho = Section(5, diameter=[5, 10, 5, 10, 5]*um, length=[10, 20, 5, 5, 10]*um)
    #morpho = Soma(20*um)
    print to_svg(morpho)


