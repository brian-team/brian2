from brian2.codegen.templates import Templater

templater = Templater('brian2.codegen.runtime.cython_rt',
                      env_globals={})

print templater.test(None, {None:['a=b']})