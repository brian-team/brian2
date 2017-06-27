
struct parameters {
    int _idx;
    double *v;
    double *vt;
    double _lio_1;
    double _lio_2;
    double _lio_3;
};

int fill_y_vector(parameters * p, double * y, int _idx)
{
  y[0] = p->vt[_idx];
  y[1] = p->v[_idx];
  return 0;
}

int empty_y_vector(parameters * p, double * y, int _idx)
{
  p->vt[_idx] = y[0];
  p->v[_idx] = y[1];
  return 0;
}

int func(double t, const double y[], double f[], void *params)
{
  struct parameters * p = (struct parameters *)params;
  int _idx = p->_idx;
  const double vt = y[0];
  const double v = y[1];
  f[0] = p->_lio_1 * (p->_lio_2 - vt);
  f[1] = p->_lio_3 * v;
  return 0;
}