'''
Statistical test for random sample generation

For a range of values of N and p does random sampling equivalent to ``rand()<p``
on the range 0 to N-1 a number of times (repeats). For each one, it computes
a 95% confidence interval of the expected number of synapses generated, which
should follow a Binomial(N, p) distribution. If it is outside this range, it
marks a fail. You would expect this to fail some percentage of the time. Since
it's a discrete distribution and not a continuous one, that percentage is not
5%, so we compute the probability q explicitly given the confidence interval.
Now, given the number of fails over the repeats of this test, we have another
Bionomial(repeats, q) and we carry out the same test, and report it if it is
outside the 95% confidence interval. Note that this can happen a few times,
but shouldn't happen systematically in one case or another. Finally, over all
the tests we carry out, we compute a mean and variance for the expected
number of test failures following a Gaussian distribution (this approximation
is OK because it's the sum of a very large number of Binomial(1, q)
random variables). We report an overall pass or fail based on the 95%
confidence interval for the total number of fails based on this distribution.

Finally, for each p we compute the distribution of intersynapse intervals
which should follow a Geometric(p) distribution. We plot this and the
expected distribution (but don't carry out a statistical test because I
couldn't be bothered to work out what the test should be).
'''

from pylab import *
from brian2 import *
from scipy.stats import binom, norm, geom

prefs.codegen.target = 'cython'
# TODO: how to do standalone in some reasonable amount of time?
N_range = [10, 100, 1000, 10000, 100000, 1000000]
repeats = 100
p_range = [0, 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 1]
alpha = 0.95 # confidence interval
numfails = 0
numchecks = 0
mu = 0
sigma2 = 0
isi = {}
isi_max = {}
num_isi = {}
for p in p_range:
    if p!=0 and p!=1:
        isi_max[p] = max(int(ceil(geom.isf(0.01, p))), 5)
        isi[p] = zeros(isi_max[p])
        num_isi[p] = 0

def check(N, p):
    global numfails, numchecks, mu, sigma2
    H = NeuronGroup(1, 'v:1', threshold='False', name='H')
    G = NeuronGroup(N, 'v:1', threshold='False', name='G')
    S = Synapses(H, G, on_pre='v+=w', name='S')
    S.connect(p=p)
    m = len(S)
    low, high = binom.interval(alpha, N, p)
    if p==0:
        low = high = 0
    elif p==1:
        low = high = N
    else:
        i = diff(S.j[:])
        i = i[i<isi_max[p]]
        b = bincount(i, minlength=isi_max[p])[:isi_max[p]]
        if b[0]:
            print 'Major error: repeated indices for N=%d, p=%.3f' % (N, p)
            raise ValueError("Repeated indices")
        isi[p] += b
        num_isi[p] += sum(b)
    q = binom.cdf(low-0.1, N, p)+binom.sf(high+0.1, N, p)
    mu += q
    sigma2 += q*(1-q)
    numchecks += 1
    if m<low or m>high:
        numfails += 1
        return True
    else:
        return False

for N in N_range:
    print 'Starting N =', N
    for p in p_range:
        num_Np_fails = 0
        num_Np_checks = 0
        for _ in xrange(repeats):
            if check(N, p):
                num_Np_fails += 1
            num_Np_checks += 1
        # work out what the failure probability is (approximately but not exactly 1-alpha
        # because it's a discrete distribution)
        low, high = binom.interval(alpha, N, p)
        if p==0:
            low = high = 0
        elif p==1:
            low = high = N
        q = binom.cdf(low-0.1, N, p)+binom.sf(high+0.1, N, p)
        low, high = binom.interval(alpha, num_Np_checks, q)
        if q==0:
            low = high = 0
        if num_Np_fails<low or num_Np_fails>high:
            print 'N=%d, p=%.3f failed %d of %d checks, outside range (%d, %d)' % (N, p, num_Np_fails,
                                                                                   num_Np_checks, low, high)
print
failrate = float(numfails)/numchecks
low, high = norm.interval(alpha, loc=mu, scale=sqrt(sigma2))
print '%d/%d=%.2f%% failed at %d%%' % (numfails, numchecks, numfails*100.0/numchecks, 100*alpha)
print 'Expected mean=%d, std dev=%d (mean fail rate=%.2f%%)' % (mu, sqrt(sigma2), 100*mu/numchecks)
if low<=numfails<=high:
    print 'Overall passed at %d%%: within range (%d, %d)' % (alpha*100, low, high)
else:
    print 'Overall failed at %d%%: outside range (%d, %d)' % (alpha*100, low, high)

figure(figsize=(10, 6))
plotnum = 0
for p in p_range:
    if p==0 or p==1:
        continue
    plotnum += 1
    subplot(2, 3, plotnum)
    n = arange(1, isi_max[p])
    plot(n, isi[p][1:], '-g', lw=3, label='Observed')
    plot(n, geom.pmf(n, p)*num_isi[p], '-k', label='Expected')
    xlabel('intersynapse interval')
    yticks([])
    title('p = %.3f' % p)
tight_layout()
show()
