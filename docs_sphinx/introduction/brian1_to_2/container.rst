Container image for Brian 1
===========================

Brian 1 depends on Python 2.x and other deprecated libraries, so running it on modern systems has become difficult. For convenience, we provide a Docker image that you can use to run existing Brian 1 code. It is based on a Debian image, and provides Brian 1.4.3, as packaged by the `NeuroDebian team <https://neuro.debian.net/>`_.
To use these images, you either need to have `docker <https://docker.com>`_,
`podman <https://podman.io/>`_ or `singularity <https://sylabs.io/singularity/>`_
installed – the commands below are shown for the ``docker`` command, but you can
simply replace them by ``podman`` if necessary. For singularity, the basic workflow
is similar but the commands are slightly different, please see the documentation.
To pull the container image with singularity, refer to
``docker://briansimulator/brian1.4.3``.

Running a graphical interface within the docker container can be complicated, and the details how to make it work depend on the host operating system. We therefore recommend to instead either 1) only use the container image to generate and save the simulation results to disk, and then to create the plots on the host system, or 2) to use the container image to plot files to disk by adding ``plt.savefig(...)`` to the script. The container already sets the matplotlib backend to ``Agg`` by default (by setting the environment variable ``MPLBACKEND``), necessary to avoid errors when no graphical interface is available.

To download the image and to rename it to ``brian1`` (for convenience only, the commands below would also work directly with the full name), use:

.. code:: shell

    docker pull docker.io/briansimulator/brian1.4.3
    docker tag briansimulator/brian1.4.3 brian1

The following command runs ``myscript.py`` with the container image providing Brian 1 and its dependencies, mapping the current directory to the working directory in the container (this means, the script has access to all files in the current directory and its subdirectories, and can also write files there):

.. code:: shell

    docker run -v "$(pwd):/workdir" brian1 python myscript.py
 
For Windows users using the Command Prompt (``cmd.exe``) instead of the Powershell, the following command will do the same thing:

.. code:: shell

    docker run -v %cd%:/workdir brian1 python myscript.py

To run an interactive ipython prompt, use:

.. code:: shell

    docker run -it -v "$(pwd):/workdir" brian1 ipython

Depending on your operating system, files written by the container might be owned by the user "root", which can lead to problems later (e.g. you cannot rename/move/delete/overwrite the file on your home system without administrator rights). On Unix-based systems, you can prevent this issue by running scripts with the same user id as the host user:

.. code:: shell

    docker run -u $(id -u):$(id -g) -v "$(pwd):/workdir" brian1 python myscript.py

Please report any issues to the `Brian discussion forum <https://brian.discourse.group>`_.

