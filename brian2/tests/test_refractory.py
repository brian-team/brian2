from brian2.equations.equations import Equations
from brian2.equations.refractory import add_refractoriness
    
def test_add_refractoriness():    
    eqs = Equations('''
    dv/dt = -x*v/second : volt (unless-refractory)
    dw/dt = -w/second : amp
    x : 1
    ''')
    # only make sure it does not throw an error
    eqs = add_refractoriness(eqs)

if __name__ == '__main__':
    test_add_refractoriness()
