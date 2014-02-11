import cPickle

import matplotlib.pyplot as plt
import numpy as np

with open('synapse_creation_times_brian1.pickle', 'r') as f:
    brian1_results = cPickle.load(f)
with open('synapse_creation_times_brian2.pickle', 'r') as f:
    brian2_results = cPickle.load(f)

results = brian1_results
results.update(brian2_results)

conditions = [('Full', True),
              ('Full (no-self)', 'i != j'),
              ('One-to-one', 'i == j'),
              ('Simple neighbourhood', 'abs(i-j) < 5'),
              ('Gauss neighbourhood', 'exp(-(i - j)**2/5) > 0.005'),
              ('Random (50%)', 0.5),
              ('Random (10%)', 0.1),
              ('Random (1%)', 0.01),
              ('Random no-self (50%)', '(i != j) * 0.5'),
              ('Random no-self (10%)', '(i != j) * 0.1'),
              ('Random no-self (1%)', '(i != j) * 0.01')]
targets = ['PythonLanguage', 'CPPLanguage', 'Brian 1']
# nicer names for the labels
lang_translation = {'PythonLanguage': 'Brian 2 (Python)',
                    'CPPLanguage': 'Brian 2 (C++)',
                    'Brian 1': 'Brian 1'}

# Do some plots
for pattern, condition in conditions:
    plt.figure()
    for lang_name in targets:
        data = [(connections, time) for ((lang, connections, p), time) in results.iteritems()
                if lang == lang_name and p == pattern]
        data.sort(key=lambda item: item[0])
        data = np.array(data).T
        plt.plot(data[0], data[1], 'o-', label=lang_translation[lang_name])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of created connections')
    plt.ylabel('time (s)')
    plt.legend(loc='best', frameon=False)
    plt.title('%s: "%s"' % (pattern, condition))
    plt.savefig('plots/%s.png' % pattern.replace(' ', '_'))