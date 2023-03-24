import inspect
import pydoc
import re
import textwrap

from sphinx.pycode import ModuleAnalyzer

from .docscrape import ClassDoc, FunctionDoc, NumpyDocString


class SphinxDocString(NumpyDocString):
    def __init__(self, docstring, config=None):
        if config is None:
            config = {}
        NumpyDocString.__init__(self, docstring, config=config)

    # string conversion routines
    @staticmethod
    def _str_header(name, symbol="`"):
        return [f".. rubric:: {name}", ""]

    @staticmethod
    def _str_field_list(name):
        return [f":{name}:"]

    @staticmethod
    def _str_indent(doc, indent=4):
        out = []
        for line in doc:
            out += [" " * indent + line]
        return out

    def _str_summary(self):
        return self["Summary"] + [""]

    def _str_extended_summary(self):
        return self["Extended Summary"] + [""]

    def _str_param_list(self, name):
        out = []
        if self[name]:
            out += self._str_field_list(name)
            out += [""]
            for param, param_type, desc in self[name]:
                out += self._str_indent([f"**{param.strip()}** : {param_type}"])
                out += [""]
                out += self._str_indent(desc, 8)
                out += [""]

        return out

    @property
    def _obj(self):
        if hasattr(self, "_cls"):
            return self._cls
        elif hasattr(self, "_f"):
            return self._f
        return None

    def _str_member_list(self):
        """
        Generate a member listing, autosummary:: table .

        """
        out = []

        for name in ["Attributes", "Methods"]:
            if not self[name]:
                continue

            out += [f".. rubric:: {name}", ""]
            prefix = getattr(self, "_name", "")

            if prefix:
                prefix = f"{prefix}."

            autosum = []
            for param, _, desc in self[name]:
                param = param.strip()
                if self._obj:
                    # Fake the attribute as a class property, but do not touch
                    # methods
                    if hasattr(self._obj, "__module__") and not (
                        hasattr(self._obj, param)
                        and callable(getattr(self._obj, param))
                    ):
                        # Do not override directly provided docstrings
                        if not len("".join(desc).strip()):
                            analyzer = ModuleAnalyzer.for_module(self._obj.__module__)
                            desc = analyzer.find_attr_docs().get(
                                (self._obj.__name__, param), ""
                            )

                        # Only fake a property if we got a docstring
                        if len("".join(desc).strip()):
                            setattr(
                                self._obj,
                                param,
                                property(lambda self: None, doc="\n".join(desc)),
                            )

                if len(prefix):
                    autosum += [f"   ~{prefix}{param}"]
                else:
                    autosum += [f"   {param}"]

            if autosum:
                out += [".. autosummary::", ""]
                out += autosum

            out += [""]
        return out

    def _str_member_docs(self, name):
        """
        Generate the full member autodocs

        """
        out = []

        if self[name]:
            prefix = getattr(self, "_name", "")

            if prefix:
                prefix += "."

            for param, _, _ in self[name]:
                if name == "Methods":
                    out += [f".. automethod:: {prefix}{param}"]
                elif name == "Attributes":
                    out += [f".. autoattribute:: {prefix}{param}"]

            out += [""]
        return out

    def _str_section(self, name):
        out = []
        if self[name]:
            out += self._str_header(name)
            out += [""]
            content = textwrap.dedent("\n".join(self[name])).split("\n")
            out += content
            out += [""]
        return out

    def _str_see_also(self, func_role):
        out = []
        if self["See Also"]:
            see_also = super()._str_see_also(func_role)
            out = [".. seealso::", ""]
            out += self._str_indent(see_also[2:])
        return out

    def _str_raises(self, name, func_role):
        if not self[name]:
            return []
        out = []
        out += self._str_header(name)
        for func, _, desc in self[name]:
            out += [f":exc:`{func}`"]
            if desc:
                out += self._str_indent([" ".join(desc)])
        out += [""]
        return out

    def _str_warnings(self):
        out = []
        if self["Warnings"]:
            out = [".. warning::", ""]
            out += self._str_indent(self["Warnings"])
        return out

    def _str_index(self):
        idx = self["index"]
        out = []
        if len(idx) == 0:
            return out

        out += [f".. index:: {idx.get('default', '')}"]
        for section, references in idx.items():
            if section == "default":
                continue
            elif section == "refguide":
                out += [f"   single: {', '.join(references)}"]
            else:
                out += [f"   {section}: {','.join(references)}"]
        return out

    def _str_references(self):
        out = []
        if self["References"]:
            out += self._str_header("References")
            if isinstance(self["References"], str):
                self["References"] = [self["References"]]
            out.extend(self["References"])
            out += [""]
            # Latex collects all references to a separate bibliography,
            # so we need to insert links to it
            out += [".. only:: latex", ""]
            items = []
            for line in self["References"]:
                m = re.match(r".. \[([a-z0-9._-]+)\]", line, re.I)
                if m:
                    items.append(m.group(1))
            out += [f"   {', '.join([f'[{item}]_' for item in items])}", ""]
        return out

    def _str_examples(self):
        return self._str_section("Examples")

    def __str__(self, indent=0, func_role="brianobj"):
        out = []
        out += self._str_index() + [""]
        out += self._str_summary()
        out += self._str_extended_summary()
        for param_list in ("Parameters", "Returns", "Other Parameters"):
            out += self._str_param_list(param_list)
        for param_list in ("Raises", "Warns"):
            out += self._str_raises(param_list, func_role)
        out += self._str_warnings()
        out += self._str_see_also(func_role)
        out += self._str_section("Notes")
        out += self._str_references()
        out += self._str_examples()
        out += self._str_member_list()
        if self["Attributes"] + self["Methods"]:
            out += [".. rubric:: Details", ""]
            for param_list in ("Attributes", "Methods"):
                out += self._str_member_docs(param_list)
        out = self._str_indent(out, indent)
        return "\n".join(out)


class SphinxFunctionDoc(SphinxDocString, FunctionDoc):
    def __init__(self, obj, doc=None, config=None):
        if config is None:
            config = {}
        FunctionDoc.__init__(self, obj, doc=doc, config=config)


class SphinxClassDoc(SphinxDocString, ClassDoc):
    def __init__(self, obj, doc=None, func_doc=None, name=None, config=None):
        if config is None:
            config = {}
        self.name = name
        ClassDoc.__init__(self, obj, doc=doc, func_doc=None, config=config)


class SphinxObjDoc(SphinxDocString):
    def __init__(self, obj, doc=None, config=None):
        if config is None:
            config = {}
        self._f = obj
        SphinxDocString.__init__(self, doc, config=config)


def get_doc_object(obj, what=None, doc=None, name=None, config=None):
    if config is None:
        config = {}
    if what is None:
        if inspect.isclass(obj):
            what = "class"
        elif inspect.ismodule(obj):
            what = "module"
        elif callable(obj):
            what = "function"
        else:
            what = "object"
    if what == "class":
        return SphinxClassDoc(
            obj, func_doc=SphinxFunctionDoc, doc=doc, name=name, config=config
        )
    elif what in ("function", "method"):
        return SphinxFunctionDoc(obj, doc=doc, config=config)
    else:
        if doc is None:
            doc = pydoc.getdoc(obj)
        return SphinxObjDoc(obj, doc, config=config)
