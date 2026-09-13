#include <stdlib.h>

long       __isoc23_strtol(const char *p, char **e, int b)   { return strtol(p, e, b); }
long long  __isoc23_strtoll(const char *p, char **e, int b)  { return strtoll(p, e, b); }
unsigned long      __isoc23_strtoul(const char *p, char **e, int b)  { return strtoul(p, e, b); }
unsigned long long __isoc23_strtoull(const char *p, char **e, int b) { return strtoull(p, e, b); }
