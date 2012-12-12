.. currentmodule:: brian2

Code generation reference
=========================

Translation
-----------

.. autofunction:: brian2.codegen.translation.make_statements
.. autofunction:: brian2.codegen.translation.translate

Language
--------

.. autoclass:: brian2.codegen.languages.base.Language

.. autoclass:: brian2.codegen.languages.base.CodeObject
    
.. autoclass:: brian2.codegen.languages.cpp.CPPLanguage
.. autoclass:: brian2.codegen.languages.cpp.CPPCodeObject
.. autofunction:: brian2.codegen.languages.cpp.c_data_type

.. autoclass:: brian2.codegen.languages.python_numexpr.NumexprPythonLanguage

Statements
----------

.. autoclass:: brian2.codegen.statements.Statement
    
Specifiers
----------

.. autoclass:: brian2.codegen.specifiers.Specifier
.. autoclass:: brian2.codegen.specifiers.Value
.. autoclass:: brian2.codegen.specifiers.ArrayVariable
.. autoclass:: brian2.codegen.specifiers.Function
.. autoclass:: brian2.codegen.specifiers.OutputVariable
.. autoclass:: brian2.codegen.specifiers.Subexpression
.. autoclass:: brian2.codegen.specifiers.Index

Templating
----------

.. autofunction:: brian2.codegen.templating.apply_code_template

User-defined functions
----------------------

.. autoclass:: brian2.codegen.functions.UserFunction
.. autoclass:: brian2.codegen.functions.SimpleUserFunction
.. autofunction:: brian2.codegen.functions.make_user_function
