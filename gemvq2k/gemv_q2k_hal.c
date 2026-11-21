#include "gemv_q2k_hal.h"
#include "gemv_q2k_uapi.h"
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <math.h>
#include <string.h>
enum { BQ8=34, BQ2K=84, TILE=4096 };
static int fd=-2;
static int dev(void){if(fd==-2)fd=open("/dev/gemv_q2k",O_RDWR|O_CLOEXEC);return fd;}
static void q8_0(const float *x, unsigned char *q, uint32_t n){
 for(uint32_t b=0;b<n/32;b++){float a=0;for(int i=0;i<32;i++){float z=fabsf(x[b*32+i]);if(z>a)a=z;}float d=a/127.f;uint16_t h=0; /* IEEE binary16 conversion for normal decode activations */
  if(d){union{float f;uint32_t u;}v={d};int e=((v.u>>23)&255)-127+15; h=(e>0&&e<31)?(uint16_t)(((v.u>>16)&0x8000)|((uint32_t)e<<10)|((v.u>>13)&1023)):0;}q[b*BQ8]=h;q[b*BQ8+1]=h>>8;
  for(int i=0;i<32;i++){int z=d?(int)lrintf(x[b*32+i]/d):0;if(z>127)z=127;if(z<-128)z=-128;q[b*BQ8+2+i]=(unsigned char)(signed char)z;}}
}
bool gemv_q2k_try(const void *w,const float *x,float *dst,uint32_t nb,uint32_t rows){
 if(!w||!x||!dst||!nb||nb>1024||dev()<0)return false; unsigned char y[1024*8*BQ8];q8_0(x,y,nb*256);
 uint32_t per_engine=(4U*1024U*1024U)/(nb*BQ2K); if(!per_engine)return false;
 for(uint32_t r=0;r<rows;){uint32_t n=rows-r, limit=per_engine*2; if(n>TILE)n=TILE;if(n>limit)n=limit;uint32_t n0=(n+1)/2,n1=n-n0;struct gemv_q2k_job j={.weights=(uint64_t)(uintptr_t)((const unsigned char*)w+(size_t)r*nb*BQ2K),.activation=(uint64_t)(uintptr_t)y,.output=(uint64_t)(uintptr_t)(dst+r),.nb=nb,.rows0=n0,.rows1=n1,.timeout_ms=1000};if(ioctl(fd,GEMV_Q2K_SUBMIT,&j))return false;r+=n;}return true;}
