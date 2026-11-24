#include "gemv_q2k_hal.h"
#include "gemv_q2k_uapi.h"
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
enum { MAX_CACHE=256, Q2K_BYTES=84 };
struct entry { const void *p; size_t bytes; unsigned handle; };
static int fd=-2, last_ok;
static struct entry cache[MAX_CACHE];
static int getfd(void){ if(fd==-2)fd=open("/dev/gemv_q2k",O_RDWR|O_CLOEXEC); return fd; }
int gemv_q2k_u96_available(void){ return getfd()>=0; }
static int handle_for(const void *p,size_t bytes,unsigned *handle){ unsigned i,freei=MAX_CACHE; struct gemv_q2k_register r;
 for(i=0;i<MAX_CACHE;i++){if(cache[i].p==p&&cache[i].bytes==bytes){*handle=cache[i].handle;return 1;}if(!cache[i].p&&freei==MAX_CACHE)freei=i;}
 if(freei==MAX_CACHE)return 0; r.address=(uint64_t)(uintptr_t)p;r.bytes=bytes;r.handle=0;if(ioctl(getfd(),GEMV_Q2K_REGISTER,&r))return 0;cache[freei]=(struct entry){p,bytes,r.handle};*handle=r.handle;return 1; }
int gemv_q2k_u96_run(const void *w,const void *a,float *o,uint32_t nb,uint32_t rows){ struct gemv_q2k_submit j;unsigned h;last_ok=0;if(!w||!a||!o||!nb||nb>64||!rows||getfd()<0)return 0;if(!handle_for(w,(size_t)nb*rows*Q2K_BYTES,&h))return 0;j=(struct gemv_q2k_submit){.handle=h,.nb=nb,.rows=rows,.timeout_ms=1000,.activation=(uint64_t)(uintptr_t)a,.output=(uint64_t)(uintptr_t)o};last_ok=ioctl(fd,GEMV_Q2K_SUBMIT,&j)==0;return last_ok;}
int gemv_q2k_u96_last_ok(void){return last_ok;}
