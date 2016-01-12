#ifndef _BRIAN_DYNAMIC_ARRAY_H
#define _BRIAN_DYNAMIC_ARRAY_H

#include<vector>

/*
 * 2D Dynamic array class
 *
 * Efficiency note: if you are regularly resizing, make sure it is the first dimension that
 * is resized, not the second one.
 *
 */
template<class T>
class DynamicArray2D
{
	int old_n, old_m;
	std::vector< std::vector<T>* > data;
public:
	int n, m;
	DynamicArray2D(int _n=0, int _m=0)
	{
		old_n = 0;
		old_m = 0;
		resize(_n, _m);
	};
	~DynamicArray2D()
	{
		resize(0, 0); // handles deallocation
	}
	void resize()
	{
		if(old_n!=n)
		{
			if(n<old_n)
			{
				for(int i=n; i<old_n; i++)
				{
					if(data[i]) delete data[i];
					data[i] = 0;
				}
			}
			data.resize(n);
			if(n>old_n)
			{
				for(int i=old_n; i<n; i++)
				{
					data[i] = new std::vector<T>;
				}
			}
			if(old_m!=m)
			{
				for(int i=0; i<n; i++)
					data[i]->resize(m);
			} else if(n>old_n)
			{
				for(int i=old_n; i<n; i++)
					data[i]->resize(m);
			}
		} else if(old_m!=m)
		{
			for(int i=0; i<n; i++)
			{
				data[i]->resize(m);
			}
		}
		old_n = n;
		old_m = m;
	};
	void resize(int _n, int _m)
	{
		n = _n;
		m = _m;
		resize();
	}
	// We cannot simply use T& as the return type here, since we don't
	// get a bool& out of a std::vector<bool>
	inline typename std::vector<T>::reference operator()(int i, int j)
	{
		return (*data[i])[j];
	}
	inline std::vector<T>& operator()(int i)
	{
		return (*data[i]);
	}
};

#endif
