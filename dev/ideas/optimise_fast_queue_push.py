from brian2 import *
from scipy import weave

code = '''
using namespace std;
const int num_queues = 100;
const int num_events = 1000000;
const int repeats = 10000;

vector<int> events;
srand((unsigned int) time(NULL));
for(int i=0; i<num_events; i++)
    events.push_back((int)(rand()%num_queues));
'''

if 1:
    code += '''
    vector< vector<int> > queue(num_queues);

    clock_t start = clock();
    for(int repeat=0; repeat<repeats; repeat++)
    {
        for(int q=0; q<num_queues; q++)
            queue[q].clear();
        //vector<int> &cq = queue[0];
        //cq.reserve(num_events);
        //cq.resize(num_events);
        //int * __restrict cqp = &(cq[0]);
        vector<int> *cq = 0;
        int cqi = -1;
        for(int i=0; i<num_events; i++)
        {
            const int q = events[i];
            if(q!=cqi)
            {
                cq = &(queue[q]);
                cqi = q;
            }
            cq->push_back(i);
            //cqp[i] = i;
            //cq.push_back(i);
            //queue[q].push_back(i);
        }
    }
    cout << "Time taken: " << (double)(clock()-start)/CLOCKS_PER_SEC;
    '''

if 0:
    code += '''
    int max_events_per_queue = 1;
    vector<int> *queue = new vector<int>(num_queues*max_events_per_queue);
    vector<int> num_events_in_queue(num_queues);

    clock_t start = clock();
    for(int repeat=0; repeat<repeats; repeat++)
    {
        for(int q=0; q<num_queues; q++) num_events_in_queue[q] = 0;
        for(int i=0; i<num_events; i++)
        {
            const int q = events[i];
            if(num_events_in_queue[q]==max_events_per_queue)
            {
                // need to do a resize operation (double max_events_per_queue)
                vector<int> *new_queue = new vector<int>(num_queues*max_events_per_queue*2);
                for(int cq=0; cq<num_queues; cq++)
                    for(int ci=0; ci<num_events_in_queue[cq]; ci++)
                        (*new_queue)[ci+cq*2*max_events_per_queue] = (*queue)[ci+cq*max_events_per_queue];
                delete queue;
                queue = new_queue;
                new_queue = 0;
                max_events_per_queue *= 2;
            }
            (*queue)[num_events_in_queue[q]+q*max_events_per_queue] = i;
            num_events_in_queue[q]++;
        }
    }
    cout << "Time taken: " << (double)(clock()-start)/CLOCKS_PER_SEC;
    delete queue;
    '''

weave.inline(code, [], compiler='msvc', headers=['<vector>'],
             extra_compile_args=['/Ox', '/EHsc', '/w', '/arch:AVX', '/fp:fast'],
             force=True)
