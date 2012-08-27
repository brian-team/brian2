from base import Language
from c import CLanguage

__all__ = ['CUDALanguage']

class CUDALanguage(CLanguage):
    def code_object(self, code):
        raise NotImplementedError
    
    def template_iterate_all(self, index, size):
        return '''
        __global__ stateupdate(int _num_neurons, double t, double dt)
        {{
            const int {index} = threadIdx.x+blockIdx.x*blockDim.x;
            if({index}>={size}) return;
            %POINTERS%
            %CODE%
        }}
        '''.format(index=index, size=size)
    
    def template_iterate_index_array(self, index, array, size):
        return '''
        __global__ stateupdate(int _num_neurons, double t, double dt)
        {{
            const int _index_{array} = threadIdx.x+blockIdx.x*blockDim.x;
            if(_index_{array}>={size}) return;
            const int {index} = {array}[_index_{array}];
            %POINTERS%
            %CODE%
        }}
        '''.format(index=index, array=array, size=size)

    def template_threshold(self):
        return '''
        __global__ threshold(int _num_neurons, double t, double dt)
        {
            const int _neuron_idx = threadIdx.x+blockIdx.x*blockDim.x;
            if(_neuron_idx>=_num_neurons) return;
            %POINTERS%
            %CODE%
            _array_cond[_neuron_idx] = _cond;
        }
        '''
        
    def template_synapses(self):
        raise NotImplementedError
    
    # TODO: optimisation of translate_statement_sequence, interleave read/write
    # accesses with computations
