.. currentmodule:: brian2

Preferences
===========

.. data:: brian_prefs

	The Brian preferences object. Access values via attributes, e.g.
	``brian_prefs.prefname``. The preferences object is of type
	:class:`~brian2.core.preferences.BrianGlobalPreferences` and new preferences must
	be defined via the method
	:meth:`~brian2.core.preferences.BrianGlobalPreferences.define`.

Built-in preferences
--------------------

.. document_brian_prefs::
 

Advanced
--------

.. autoclass:: brian2.core.preferences.BrianGlobalPreferences

    