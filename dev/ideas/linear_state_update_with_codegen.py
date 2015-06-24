from brian import *
from scipy import weave
import time
import numexpr

N = 10000

eqs = '''
dv/dt = (ge+gi-(v+49*mV))/(10*ms) : volt
dge/dt = -ge/(5*ms) : volt
dgi/dt = -gi/(10*ms) : volt
'''

P = NeuronGroup(N, eqs)

Scopy = P._S.copy()

run(20*ms)

B = P._state_updater.B
A = P._state_updater.A
C = P._state_updater._C

print 'A =\n'+str(A)
print 'C =', C[:, 0]

print 'Verification (these should all be the same)'

print P._S[:, 0], 'with brian'

def withdot(nsteps):
    for _ in xrange(nsteps):
        S[:] = dot(A, S)
    #    dot(A, S, out=S) # doesn't seem to work
    #    # but this does work (no quicker though)
    #    dot(A, S, out=T)
    #    S[:] = T
        add(S, C, S)

S = Scopy.copy()
withdot(200)
print S[:, 0], 'with dot'

Atensor = repeat(A.reshape((3, 3, 1)), N, axis=2)
Ctensor = repeat(C.reshape((3, 1)), N, axis=1)
def withtensordot(nsteps):
    for _ in xrange(nsteps):
        #S[:] = tensordot(Atensor, S.reshape((1, 3, N)), axes=1)
        S[:] = sum(Atensor*S.reshape((1, 3, N)), axis=1)
    #    dot(A, S, out=S) # doesn't seem to work
    #    # but this does work (no quicker though)
    #    dot(A, S, out=T)
    #    S[:] = T
        add(S, Ctensor, S)

S = Scopy.copy()
withtensordot(200)
print S[:, 0], 'with tensordot'

def witheinsum(nsteps):
    for _ in xrange(nsteps):
        S[:] = einsum('jk...,k...->j...', Atensor, S)
        add(S, Ctensor, S)

S = Scopy.copy()
witheinsum(200)
print S[:, 0], 'with einsum'

def withcopydot(nsteps):
    v, ge, gi = S    
    T = vstack((v, ge, gi))
    for _ in xrange(nsteps):
        v_next, ge_next, gi_next = dot(A, T)+C
        v[:] = v_next
        ge_next[:] = ge_next
        gi_next[:] = gi_next

S = Scopy.copy()
withdot(200)
print S[:, 0], 'with copydot'

codeweave = '''
double *v = S;
double *ge = S+N;
double *gi = S+2*N;
for(int i=0; i<N; ++i)
{
    double v_next = A[0]*v[i]+A[1]*ge[i]+A[2]*gi[i]+C[0];
    double ge_next = A[3]*v[i]+A[4]*ge[i]+A[5]*gi[i]+C[1];
    double gi_next = A[6]*v[i]+A[7]*ge[i]+A[8]*gi[i]+C[2];
    v[i] = v_next;
    ge[i] = ge_next;
    gi[i] = gi_next;
}
'''

def withweave(nsteps):
    for _ in xrange(nsteps):
        weave.inline(codeweave, ['S', 'N', 'A', 'C'],
                     compiler='gcc',
                     extra_compile_args=['-O3', '-ffast-math',
                                         '-march=native',
                                         ])

S = Scopy.copy()
withweave(200)
print S[:, 0], 'with weave'

codeweavemulti = '''
double *v = S;
double *ge = S+N;
double *gi = S+2*N;
for(int i=0; i<N; ++i)
{
    double v_next = A00[i]*v[i]+A01[i]*ge[i]+A02[i]*gi[i]+C0[i];
    double ge_next = A10[i]*v[i]+A11[i]*ge[i]+A12[i]*gi[i]+C1[i];
    double gi_next = A20[i]*v[i]+A21[i]*ge[i]+A22[i]*gi[i]+C2[i];
    v[i] = v_next;
    ge[i] = ge_next;
    gi[i] = gi_next;
}
'''

A00 = A[0, 0]*ones(N)
A01 = A[0, 1]*ones(N)
A02 = A[0, 2]*ones(N)
A10 = A[1, 0]*ones(N)
A11 = A[1, 1]*ones(N)
A12 = A[1, 2]*ones(N)
A20 = A[2, 0]*ones(N)
A21 = A[2, 1]*ones(N)
A22 = A[2, 2]*ones(N)
C0 = C[0]*ones(N)
C1 = C[1]*ones(N)
C2 = C[2]*ones(N)
def withweavemulti(nsteps):
    for _ in xrange(nsteps):
        weave.inline(codeweavemulti, ['S', 'N', 'A00', 'A01', 'A02', 'A10',
                                      'A11', 'A12', 'A20', 'A21', 'A22',
                                      'C0', 'C1', 'C2'],
                     compiler='gcc',
                     extra_compile_args=['-O3', '-ffast-math',
                                         '-march=native',
                                         ])

S = Scopy.copy()
withweavemulti(200)
print S[:, 0], 'with weavemulti'

codeweaveopt = '''
double *v = S;
double *ge = S+N;
double *gi = S+2*N;
for(int i=0; i<N; ++i)
{
    double v_next = A[0]*v[i]+A[1]*ge[i]+A[2]*gi[i]+C[0];
    double ge_next = A[4]*ge[i];
    double gi_next = A[8]*gi[i];
    v[i] = v_next;
    ge[i] = ge_next;
    gi[i] = gi_next;
}
'''

def withweaveopt(nsteps):
    for _ in xrange(nsteps):
        weave.inline(codeweaveopt, ['S', 'N', 'A', 'C'],
                     compiler='gcc',
                     extra_compile_args=['-O3', '-ffast-math',
                                         '-march=native',
                                         ])

S = Scopy.copy()
withweaveopt(200)
print S[:, 0], 'with weaveopt'

codeweaveopt2 = '''
double *v = S;
double *ge = S+N;
double *gi = S+2*N;
for(int i=0; i<N; ++i)
{
    double v_next = A[0]*v[i]+A[1]*ge[i]+A[2]*gi[i]+C[0];
    double ge_next = A[4]*ge[i];
    double gi_next = A[8]*gi[i];
    v[i] = v_next;
    ge[i] = ge_next;
    gi[i] = gi_next;
}
'''.replace('A[0]', repr(A[0, 0]))\
   .replace('A[1]', repr(A[0, 1]))\
   .replace('A[2]', repr(A[0, 2]))\
   .replace('A[4]', repr(A[1, 1]))\
   .replace('A[8]', repr(A[2, 2]))\
   .replace('C[0]', repr(C[0, 0]))

def withweaveopt2(nsteps):
    for _ in xrange(nsteps):
        weave.inline(codeweaveopt2, ['S', 'N', 'A', 'C'],
                     compiler='gcc',
                     extra_compile_args=['-O3', '-ffast-math',
                                         '-march=native',
                                         ])

S = Scopy.copy()
withweaveopt2(200)
print S[:, 0], 'with weaveopt2'

def withpython(nsteps):
    v = S[0, :]
    ge = S[1, :]
    gi = S[2, :]
    for _ in xrange(nsteps):
        v_next = A[0, 0]*v+A[0, 1]*ge+A[0, 2]*gi+C[0]
        ge_next = A[1, 0]*v+A[1, 1]*ge+A[1, 2]*gi+C[1]
        gi_next = A[2, 0]*v+A[2, 1]*ge+A[2, 2]*gi+C[2]
        v[:] = v_next
        ge[:] = ge_next
        gi[:] = gi_next

S = Scopy.copy()
withpython(200)
print S[:, 0], 'with python'

def withnumexpr(nsteps):
    v = S[0, :]
    ge = S[1, :]
    gi = S[2, :]
    A00, A01, A02 = A[0]
    A10, A11, A12 = A[1]
    A20, A21, A22 = A[2]
    C0, C1, C2 = C
    for _ in xrange(nsteps):
        v_next = numexpr.evaluate('A00*v+A01*ge+A02*gi+C0')
        ge_next = numexpr.evaluate('A10*v+A11*ge+A12*gi+C1')
        gi_next = numexpr.evaluate('A20*v+A21*ge+A22*gi+C2')
        v[:] = v_next
        ge[:] = ge_next
        gi[:] = gi_next

S = Scopy.copy()
withnumexpr(200)
print S[:, 0], 'with numexpr'

# verified that it works, now do speed test

numsteps = 100000000/N
if N<=100: numsteps /= 10
if N<=10: numsteps /= 10

print
print 'N:', N
print 'numsteps:', numsteps

def timespec(t):
    if t>tref:
        return '%.2f (%.1fx slower)'%(t, t/tref)
    else:
        return '%.2f (%.1fx faster)'%(t, tref/t)

start = time.time()
withdot(numsteps)
tref = t = time.time()-start
print 'With dot: %.2f'%t

def dotiming(func):
    start = time.time()
    func(numsteps)
    t = time.time()-start
    print 'With', func.__name__[4:]+':', timespec(t)

dotiming(withcopydot)
#dotiming(withtensordot)
dotiming(witheinsum)
dotiming(withweave)
dotiming(withweavemulti)
#dotiming(withweaveopt)
dotiming(withweaveopt2)
dotiming(withpython)
#dotiming(withnumexpr)
