Targets
=======
Code is generated for a specific target.
Here is a list of potential targets:

* Python (more precisely, numpy)
* C
* GPU
* Android
	Android is based on linux and uses a Java API.
	But there also seems to be something called Renderscript for numerical
	computation based on a C-like language.
	http://www.slideshare.net/yrameshrao/android-understanding-6453986
	[Dan] Renderscript is for the GPU I think. There is also NDK for
	developing in languages such as C and C++ on Android, but then it is
	more model-dependent (whereas Renderscript is model independent I think).
* Cluster or multicores (MPI)
* Spinnaker
	[Dan] I explored this a bit with the Spinnaker people, and the tricky
	thing is that they use fixed point arithmetic, which implies that you
	really need to write the state update step with that in mind or you will
	definitely get junk results. I had a little look, and I couldn't find any
	work on automatically rewriting computations to work in fixed arithmetic,
	but I'd be surprised if there really wasn't any. Anyway, this would be a
	prerequisite for making Spinnaker work I think.
* Other neuromorphic chips
* C++ simulator
* Robot (i.e., code to be run on a specific robot)
* PyNN
* PyNeuron
* NineML

What should one do to develop support for a specific target?

* Code generation. Translation to target code could be identical for several targets
  (e.g. that use C or C-like code). Decoration (adding loops etc) is most likely
  target-specific.
* Managing device memory.
* SpikeQueue and possibly a few other key classes.
* Target-specific library. This is code with which the generated will be compiled.
  This is used mostly when creating a complete code.
