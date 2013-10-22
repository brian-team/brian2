#ifndef _BRIAN_DYNAMIC_ARRAY_H
#define _BRIAN_DYNAMIC_ARRAY_H

#include<vector>

using namespace std;

template<class T>
class DynamicArray2D
{
public:
	int n, m;
	vector< vector<T> > data;
	DynamicArray2D(int _n=0, int _m=0)
	{
		resize(_n, _m);
	};
	void resize()
	{
		data.resize(n);
		for(int i=0; i<n; i++)
		{
			data[i].resize(m);
		}
	};
	void resize(int _n, int _m)
	{
		n = _n;
		m = _m;
		resize();
	}
	inline T& operator()(int i, int j)
	{
		return data[i][j];
	}
};

#endif
