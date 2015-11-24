'''
Code for analysing string conditions into multiple components for optimisation

Start with an expression expr that is assumed to return a boolean value. We analyse
this expression into conjunctive or disjunctive normal form, and replace all, e.g.
not a>b with a<=b:

Consider:     (i<5 or i>10) and rand()<p                  Conjunctive normal form (CNF)
equivalently: (i<5 and rand()<p) or (i>10 and rand()<p)   Disjunctive normal form (DNF)

Another one:  (i<5 and rand()<p1) or (i>=5 and rand()<p2)         DNF
              (A&B)|(C&D)
              (A|(C&D))&(B|(C&D))
              (A|C)&(A|D)&(B|C)&(B|D)
              (i<5 or i>=5) and (i<5 or rand()<p2) and (rand()<p1 or i>=5) and (rand()<p1 or rand()<p2)
              (i<5 or rand()<p2) and (rand()<p1 or i>=5) and (rand()<p1 or rand()<p2)         (CNF)

Suggests that DNF is more useful.

Process:
For each j:
  Create empty set I of i values that pass the test
    For each subexpr in subexpr1 or subexpr2 or subexpr3 or ...
        subexpr is atom1 and atom2 and ...
        Analyse atoms into the following forms where possible:
        1. i < f(j), etc.
        2. rand() < p(j)
        Compute imin and imax from atoms of type 1
        If a term of type 2 is present, efficiently construct a candidate set from [imin, imax]
        Either iterate i from imax to imax or over the candidate set:
            Check remaining conditions, add i to I
'''
import ast
from collections import OrderedDict

from sympy.logic.boolalg import to_dnf

from brian2.parsing.rendering import NodeRenderer
from brian2.utils.stringtools import get_identifiers, indent
from brian2.parsing.sympytools import sympy_to_str, str_to_sympy

class BooleanAtomRenderer(NodeRenderer):
    def __init__(self):
        self.atoms = OrderedDict()
        self.name_to_atom = OrderedDict()
        self.n = 0
        super(BooleanAtomRenderer, self).__init__(use_vectorisation_idx=False)
    def render_node(self, node):
        if node.__class__.__name__!='BoolOp':
            if node in self.atoms:
                atomname = self.atoms[node]
            else:
                atomname = '_atom_'+str(self.n)
                self.n += 1
                self.atoms[node] = atomname
                self.name_to_atom[atomname] = node
            return atomname
        else:
            return super(BooleanAtomRenderer, self).render_node(node)

def analyse(expr):#, variables):
    orignode = ast.parse(expr, mode='eval').body
    # we render all the nodes in the tree into separate atoms because
    # if there is something like (a or b) and rand()<p then it will
    # get converted to DNF (a and rand()<p) or (b and rand()<p), i.e.
    # the rand() is computed twice, and we want it to be computed
    # only once.
    bar = BooleanAtomRenderer()
    newexpr = bar.render_node(orignode)
    # Now convert this newexpr which should consist only of atom names, 'and' and 'or'
    # into disjunctive normal form with sympy, and convert back to a string
    newexpr = sympy_to_str(to_dnf(str_to_sympy(newexpr)))
    # now we use the AST representation to analyse this into its DNF form,
    # the output dnf_form is a list of lists of strings. The outermost lists consist
    # of a specification of the conditions that are combined with or, and
    # the innermost lists consists of the atoms that are combined with and.
    dnfnode = ast.parse(newexpr, mode='eval').body
    if dnfnode.__class__.__name__=='Name':
        # consider the case where is only one atom (e.g. rand()<p)
        dnf_form = [[dnfnode.id]]
    elif dnfnode.op.__class__.__name__=='And':
        # consider the case where there are no ors, just a single and
        dnf_form = [[n.id for n in dnfnode.values]]
    elif dnfnode.op.__class__.__name__=='Or':
        # usual case, the outermost node is an Or, so the innermost ought
        # to be And or names
        dnf_form = []
        for subnode in dnfnode.values:
            if subnode.__class__.__name__=='BoolOp':
                if subnode.op.__class__.__name__!='And':
                    raise ValueError("Something went wrong")
                dnf_form.append([n.id for n in subnode.values])
            elif subnode.__class__.__name__=='Name':
                dnf_form.append([subnode.id])
            else:
                raise ValueError("Something went wrong")
    else:
        raise ValueError("Something went wrong") # TODO: improve this message, but it shouldn't happen
    # now analyse each atom
    for atomnode, atomname in bar.atoms.items():
        analyse_atom(atomnode)
    # produce final analysed version
    conditions = []
    for andform in dnf_form:
        i_inequalities = []
        rand = None
        remaining_terms = []
        for atomname in andform:
            atomnode = bar.name_to_atom[atomname]
            if hasattr(atomnode, 'atom_type'):
                if atomnode.atom_type=='i_inequality':
                    i_inequalities.append((atomnode.atom_op, atomnode.atom_rhs))
                    continue
                if atomnode.atom_type=='rand' and rand is None:
                    rand = atomnode.atom_probability
                    continue
            remaining_terms.append(atomname)
        if len(remaining_terms):
            nr = NodeRenderer(use_vectorisation_idx=False)
            remlines = ['%s = %s' % (atomname, nr.render_node(bar.name_to_atom[atomname])) for atomname in remaining_terms]
            remlines.append('_cond = '+' and '.join(atomname for atomname in remaining_terms))
            remaining_code = '\n'.join(remlines)
        else:
            remaining_code = None
        conditions.append((i_inequalities, rand, remaining_code))

    # # DEBUG: print it all out
    # for atomnode, atomname in bar.atoms.items():
    #     print atomname, '=', NodeRenderer(use_vectorisation_idx=False).render_node(atomnode),
    #     if hasattr(atomnode, 'atom_type'):
    #         print ':', atomnode.atom_type,
    #         if atomnode.atom_type=='i_inequality':
    #             print atomnode.atom_op, atomnode.atom_rhs,
    #         elif atomnode.atom_type=='rand':
    #             print atomnode.atom_probability,
    #     print
    # print
    # print newexpr
    # print
    # print 'OR:'
    # for andform in dnf_form:
    #     print '    AND: '+', '.join(andform)
    # print
    # print 'CONDITIONS'
    # for (i_inequalities, rand, remaining_code) in conditions:
    #     print '    Condition'
    #     print '        Inequalities on i:', i_inequalities
    #     print '        Random probability:', rand
    #     if remaining_code:
    #         print '        Remaining code:\n'+indent(remaining_code, numtabs=3)

    return conditions


# Relationship between operators if you swap LHS op RHS
reverse_op = {'<': '>',
              '<=': '>=',
              '>': '<',
              '>=': '<=',
              '==': '==',
              #'!=': '!=', # not supported (has no computational advantage)
              }


def analyse_atom(node):
    if node.__class__.__name__!='Compare':
        return
    if len(node.comparators)>1:
        return
    lhs = node.left
    rhs = node.comparators[0]
    op = NodeRenderer.expression_ops[node.ops[0].__class__.__name__]
    if op=='!=':
        return # != has no computational advantage as it only excludes a single value
    # crude initial analysis, check directly for known comparison types
    nr = NodeRenderer(use_vectorisation_idx=False)
    lhs_expr = nr.render_node(lhs).strip()
    lhs_idents = get_identifiers(lhs_expr)
    rhs_expr = nr.render_node(rhs).strip()
    rhs_idents = get_identifiers(rhs_expr)
    if lhs_expr=='i' and 'i' not in rhs_idents:
        node.atom_type = 'i_inequality'
        node.atom_op = op
        node.atom_rhs = rhs_expr
        return
    elif rhs_expr=='i' and 'i' not in lhs_idents:
        node.atom_type = 'i_inequality'
        node.atom_op = reverse_op[op]
        node.atom_rhs = lhs_expr
        return
    elif lhs_expr=='rand()' and op in ('<', '<=') and 'i' not in rhs_idents:
        node.atom_type = 'rand'
        node.atom_probability = rhs_expr
        return
    elif rhs_expr=='rand()' and op in ('>', '>=') and 'i' not in lhs_idents:
        node.atom_type = 'rand'
        node.atom_probability = lhs_expr
        return


def numpy_test_code(expr, N, duration=1):
    import jinja2, time
    from brian2.parsing.rendering import NumpyNodeRenderer
    expr = NodeRenderer().render_expr(expr)
    code = '''
from numpy import *
from numpy.random import rand as _rand
def rand(x):
    return _rand(N)
all_i = []
all_j = []
_vectorisation_idx = arange(N)
for j in xrange(N):
    i = _vectorisation_idx
    _cond = {{expr}}
    i = _cond.nonzero()[0]
    all_i.append(i)
    all_j.append(full(len(i), j, dtype=int))
all_i = hstack(all_i)
all_j = hstack(all_j)
    '''
    code = jinja2.Template(code).render(expr=NumpyNodeRenderer().render_expr(expr))
    # print code
    # exit()
    code = compile(code, '', 'exec')
    ns = {'N': N}
    start = time.time()
    completed = 0
    while time.time()-start<duration:
        exec code in ns
        completed += 1
        # print len(ns['all_i']), len(ns['all_j'])
        # break
    return (time.time()-start)/completed


def optimised_numpy_test_code(expr, N, duration=1):
    import jinja2, time
    from brian2.parsing.rendering import NumpyNodeRenderer
    #expr = NodeRenderer().render_expr(expr)
    conditions = analyse(expr)
    code = '''
from numpy import *
from numpy.random import choice, binomial, rand
from sklearn.utils.random import sample_without_replacement
from random import sample
from __builtin__ import max
all_i = []
all_j = []
_vectorisation_idx = arange(N)
for j in xrange(N):
    i = _vectorisation_idx

    cur_i = []
    {% for (i_inequalities, rand, remaining_terms) in conditions %}
    imins = (0,
    {% for op, rhs in i_inequalities %}
        {% if op=='>=' or op=='==' %}
        int(ceil({{ rhs }})),
        {% elif op=='>' %}
        1+int(floor({{ rhs }})),
        {% endif %}
    {% endfor %}
        )
    imaxes = (N-1,
    {% for op, rhs in i_inequalities %}
        {% if op=='<=' or op=='==' %}
        int(floor({{ rhs }})),
        {% elif op=='<' %}
        -1+int(ceil({{ rhs }})),
        {% endif %}
    {% endfor %}
        )
    imin = max(imins)
    imax = min(imaxes)
    if imax>=imin:
        i = _vectorisation_idx[imin:imax+1]

        {% if rand %}
        p = {{rand}}
        if p>1.0:
            p = 1.0
        k = binomial(imax-imin+1, p)
        i = i[sample_without_replacement(len(i), k)] # fast
        #i = sample(i, k) # ok
        #i = choice(i, k, replace=False) # very slow
        {% endif %}

        {% if remaining_code %}
        {{ indent(remaining_code, numtabs=1) }}
        i, = _cond.nonzero()
        {% endif %}

        cur_i.append(i)
    else:
        cur_i.append([])

    {% endfor %}

    {% if len(conditions)>1 %}
    i = unique(hstack(cur_i))
    {% endif %}

    all_i.append(i)
    all_j.append(full(len(i), j, dtype=int))
all_i = hstack(all_i)
all_j = hstack(all_j)
    '''
    code = jinja2.Template(code).render(conditions=conditions, len=len)
    # print code
    # exit()
    code = compile(code, '', 'exec')
    # return
    ns = {'N': N}
    start = time.time()
    completed = 0
    while time.time()-start<duration:
        exec code in ns
        completed += 1
        # print len(ns['all_i']), len(ns['all_j'])
        # break
    return (time.time()-start)/completed

if __name__=='__main__':
    import pylab
    from numpy import array
    #analyse('(i<5 or i>=10) and rand()<p')
    #analyse('i<5 and rand()<p')
    #analyse('i<5 or rand()<p')
    #optimised_numpy_test_code('(i<5 or i>=10) and rand()<0.1', 1000)
    #expr = '(i<5 or i>=10) and rand()<0.01'
    expr = 'i>j-100 and i<j+100 and rand()<0.1'
    #expr = 'rand()<0.1'
    #expr = 'i==j'
    #expr = 'i>j-10 and i<j+10'
    #expr = '(i<j-100 and rand()<0.01) or (i>j+100 and rand()<0.01)'
    N_range = [10, 100, 1000, 10000,
               #100000,
               ]
    all_t_optim = []
    all_t_std = []
    for N in N_range:
        print 'Doing', N
        t_optim = optimised_numpy_test_code(expr, N)
        # print 'N =', N
        # print 'Optimised:', t_optim
        t_std = numpy_test_code(expr, N)
        # print 'Standard:', t_std
        # print 'Optimisation improvement: %.1f x' % (t_std/t_optim)
        all_t_optim.append(t_optim)
        all_t_std.append(t_std)
    pylab.figure(figsize=(14, 5))
    pylab.subplot(121)
    pylab.loglog(N_range, all_t_optim, label='Optimised')
    pylab.loglog(N_range, all_t_std, label='Standard')
    pylab.xlabel('N')
    pylab.ylabel('Time (s)')
    pylab.legend(loc='best')
    pylab.title(expr)
    pylab.subplot(122)
    pylab.semilogx(N_range, array(all_t_std)/array(all_t_optim), label='Optimised')
    pylab.axhline(1, ls='--', c='k')
    pylab.xlabel('N')
    pylab.ylabel('Improvement')
    pylab.tight_layout()
    pylab.show()
