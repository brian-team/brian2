#ifndef _BRIAN_NAMESPACE_H
#define _BRIAN_NAMESPACE_H

#include<unordered_map>
#include<string>

class brian_namespace
{
public:
	std::unordered_map<string, void*> ns_map;
	template<class T>
	void set(string key, T &val)
	{
		ns_map[key] = (void*)&val;
	}
	template<class T>
	T& get(string key)
	{
		return *((T*)ns_map[key]);
	}
};

#endif
