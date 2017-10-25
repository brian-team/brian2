Releasing a new version of Brian
================================

TODO: This needs more info about the basic process

Authentification tokens
-----------------------
The test servers will automatically upload new conda packages to our channel
at anaconda.org. To do this, ``travis.yml`` and ``appveyor.yml`` contain an
encrypted version of an authentification token. To generate a token, you need
to be a member of the *brian-team* organization and have the ``anaconda-client``
package installed (alternatively, you can create a token on the website).

To create the token, run:

.. code:: console

    anaconda auth -c -o brian-team -n brian-team-token -s "repos conda api"

.. warning::

    Do not share the generated token, it servers as a username + password
    replacement and could be used to upload/delete/modify packages in our
    channel.

Now, encrypt the generated token for inclusing in ``travis.yml`` and
``appveyor.yml``.

Encryption for travis
~~~~~~~~~~~~~~~~~~~~~

More information: https://docs.travis-ci.com/user/encryption-keys/

First, install the travis CLI tool, if you do not already have it.

.. code:: console

    gem install travis

Then, navigate into your ``brian2`` working copy (i.e. your checked out git
repository), and run:

.. code:: console

    travis encrypt BINSTAR_TOKEN="...your token..."

Copy the returned ``secure: ....`` line into ``travis.yml`` (into the
``env: global`` section at the top).

Encryption for appveyor
~~~~~~~~~~~~~~~~~~~~~~~

Log into appveyor using the ``brianteam`` team account and navigate to the
"Encrypt data" website (will automatically ask you to log in if you are not):
https://ci.appveyor.com/tools/encrypt

Paste in the token returned by ``anaconda auth`` earlier (just the token, not
``BINSTAR_TOKEN=...``)

Add the encrypted value to ``appveyor.yml`` (into the
``environment: BINSTAR_TOKEN`` section at the top).
