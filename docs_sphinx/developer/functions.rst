Adding support for new functions
================================

For a description of Brian's function system from the user point of view, see
:doc:`../advanced/functions`.

The default functions available in Brian are stored in the `DEFAULT_FUNCTIONS`
dictionary. New `Function` objects can be added to this dictionary to make them
available to all Brian code, independent of its namespace.

To add a new implementation for a code generation target, a
`FunctionImplementation` can be added to the `Function.implementations`
dictionary. The key for this dictionary has to be either a `CodeGenerator` class
object, or a `CodeObject` class object. The `CodeGenerator` of a `CodeObject`
(e.g. `CPPCodeGenerator` for `CPPStandaloneCodeObject`) is used as a fallback if no
implementation specific to the `CodeObject` class exists.

If a function is already provided for the target language (e.g. it is part of
a library imported by default), using the same name, all that is needed is to
add an empty `FunctionImplementation` object to mark the function as
implemented. For example, ``exp`` is a standard function in C++::

        DEFAULT_FUNCTIONS['exp'].implementations[CPPCodeGenerator] = FunctionImplementation()

Some functions are implemented but have a different name in the target language.
In this case, the `FunctionImplementation` object only has to specify the new
name::

    DEFAULT_FUNCTIONS['arcsin'].implementations[CPPCodeGenerator] = FunctionImplementation('asin')

Finally, the function might not exist in the target language at all, in this
case the code for the function has to be provided, the exact form of this
code is language-specific. In the case of C++, it's a dictionary of code blocks::

    clip_code = {'support_code': '''
            double _clip(const float value, const float a_min, const float a_max)
            {
	            if (value < a_min)
	                return a_min;
	            if (value > a_max)
	                return a_max;
	            return value;
            }
            '''}
    DEFAULT_FUNCTIONS['clip'].implementations[CPPCodeGenerator] = FunctionImplementation('_clip',
                                                                                    code=clip_code)
