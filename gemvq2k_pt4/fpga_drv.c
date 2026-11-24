// Ultra96-V2 Q2_K driver: pin model pages once; no token-time weight copies.
#include <linux/module.h>
#include <linux/platform_device.h>
#include <linux/of.h>
#include <linux/miscdevice.h>
#include <linux/fs.h>
#include <linux/io.h>
#include <linux/dma-mapping.h>
#include <linux/mm.h>
#include <linux/uaccess.h>
#include <linux/mutex.h>
#include <linux/jiffies.h>
#include "gemv_q2k_uapi.h"
#define NAME "gemv_q2k"
#define MAX_REG 256
#define MAX_BLOCKS 64
#define MAX_ROWS 8192
#define ACT_BYTES (MAX_BLOCKS*292)
#define OUT_BYTES (MAX_ROWS*4)
enum { CTRL=0, PT=0x10, ACT=0x20, OUT=0x30, WOFF=0x40, START=0x50, NB=0x60, ROWS=0x70, STATUS=0x80 };
struct regn { u64 bytes; u32 page_offset; unsigned npages; struct page **pages; dma_addr_t *dma; u64 *pt_cpu; dma_addr_t pt_dma; };
struct gemv { struct device *dev; void __iomem *r; struct miscdevice misc; struct mutex lock; struct regn reg[MAX_REG]; void *act,*out; dma_addr_t act_dma,out_dma; };
static inline u32 rd(struct gemv*g,u32 o){return readl(g->r+o);} static inline void wr(struct gemv*g,u32 o,u32 v){writel(v,g->r+o);} static void ptr(struct gemv*g,u32 o,dma_addr_t a){wr(g,o,lower_32_bits(a));wr(g,o+4,upper_32_bits(a));}
static void release_reg(struct gemv*g,struct regn*x){unsigned i;if(!x->pages)return;for(i=0;i<x->npages;i++){if(x->dma[i])dma_unmap_page(g->dev,x->dma[i],PAGE_SIZE,DMA_TO_DEVICE);put_page(x->pages[i]);}dma_free_coherent(g->dev,x->npages*sizeof(u64),x->pt_cpu,x->pt_dma);kfree(x->dma);kfree(x->pages);memset(x,0,sizeof(*x));}
static long reg_weight(struct gemv*g,struct gemv_q2k_register __user*u){struct gemv_q2k_register q;struct regn*x;unsigned i,n,slot;unsigned long first;int got;if(copy_from_user(&q,u,sizeof(q)))return-EFAULT;if(!q.bytes||q.bytes>(u64)4*1024*1024*1024)return-EINVAL;mutex_lock(&g->lock);for(slot=0;slot<MAX_REG&&g->reg[slot].pages;slot++);if(slot==MAX_REG){mutex_unlock(&g->lock);return-ENOSPC;}x=&g->reg[slot];first=(unsigned long)q.address&PAGE_MASK;n=DIV_ROUND_UP(((unsigned long)q.address&(PAGE_SIZE-1))+q.bytes,PAGE_SIZE);x->pages=kcalloc(n,sizeof(*x->pages),GFP_KERNEL);x->dma=kcalloc(n,sizeof(*x->dma),GFP_KERNEL);if(!x->pages||!x->dma)goto oom;got=get_user_pages_fast(first,n,0,x->pages);if(got!=n){if(got>0){x->npages=got;release_reg(g,x);}goto fault;}x->pt_cpu=dma_alloc_coherent(g->dev,n*sizeof(u64),&x->pt_dma,GFP_KERNEL);if(!x->pt_cpu)goto oom;x->npages=n;x->bytes=q.bytes;x->page_offset=(unsigned long)q.address&(PAGE_SIZE-1);for(i=0;i<n;i++){x->dma[i]=dma_map_page(g->dev,x->pages[i],0,PAGE_SIZE,DMA_TO_DEVICE);if(dma_mapping_error(g->dev,x->dma[i]))goto oom;x->pt_cpu[i]=x->dma[i];dma_sync_single_for_device(g->dev,x->dma[i],PAGE_SIZE,DMA_TO_DEVICE);}q.handle=slot+1;mutex_unlock(&g->lock);return copy_to_user(u,&q,sizeof(q))?-EFAULT:0;oom:release_reg(g,x);mutex_unlock(&g->lock);return-ENOMEM;fault:mutex_unlock(&g->lock);return-EFAULT;}
static long submit(struct gemv*g,struct gemv_q2k_submit __user*u){struct gemv_q2k_submit q;struct regn*x;size_t ab,ob;unsigned long end;int rc=0;if(copy_from_user(&q,u,sizeof(q)))return-EFAULT;if(!q.handle||q.handle>MAX_REG||!q.nb||q.nb>MAX_BLOCKS||!q.rows||q.rows>MAX_ROWS)return-EINVAL;ab=(size_t)q.nb*292;ob=(size_t)q.rows*4;mutex_lock(&g->lock);x=&g->reg[q.handle-1];if(!x->pages||((u64)q.start_block+(u64)q.rows*q.nb)*84>x->bytes){rc=-EINVAL;goto out;}if(copy_from_user(g->act,(void __user *)(unsigned long)q.activation,ab)){rc=-EFAULT;goto out;}ptr(g,PT,x->pt_dma);ptr(g,ACT,g->act_dma);ptr(g,OUT,g->out_dma);wr(g,WOFF,x->page_offset);wr(g,START,q.start_block);wr(g,NB,q.nb);wr(g,ROWS,q.rows);wr(g,CTRL,1);end=jiffies+msecs_to_jiffies(q.timeout_ms?:1000);while(!(rd(g,CTRL)&2))if(time_after(jiffies,end)){rc=-ETIMEDOUT;goto out;}q.status=rd(g,STATUS);if(q.status&4){rc=-EIO;goto out;}if(copy_to_user((void __user *)(unsigned long)q.output,g->out,ob))rc=-EFAULT;out:mutex_unlock(&g->lock);return !rc&&copy_to_user(u,&q,sizeof(q))?-EFAULT:rc;}
static long ioctl_(struct file*f,unsigned c,unsigned long a){struct gemv*g=container_of(f->private_data,struct gemv,misc);return c==GEMV_Q2K_REGISTER?reg_weight(g,(void __user*)a):c==GEMV_Q2K_SUBMIT?submit(g,(void __user*)a):-ENOTTY;}static const struct file_operations fops={.owner=THIS_MODULE,.unlocked_ioctl=ioctl_};
static int probe(struct platform_device*p){struct gemv*g;int rc;g=devm_kzalloc(&p->dev,sizeof(*g),GFP_KERNEL);if(!g)return-ENOMEM;g->dev=&p->dev;mutex_init(&g->lock);g->r=devm_platform_ioremap_resource(p,0);if(IS_ERR(g->r))return PTR_ERR(g->r);g->act=dma_alloc_coherent(g->dev,ACT_BYTES,&g->act_dma,GFP_KERNEL);g->out=dma_alloc_coherent(g->dev,OUT_BYTES,&g->out_dma,GFP_KERNEL);if(!g->act||!g->out)return-ENOMEM;g->misc=(struct miscdevice){.minor=MISC_DYNAMIC_MINOR,.name=NAME,.fops=&fops};rc=misc_register(&g->misc);if(rc)return rc;platform_set_drvdata(p,g);return 0;}static int remove_(struct platform_device*p){struct gemv*g=platform_get_drvdata(p);int i;misc_deregister(&g->misc);for(i=0;i<MAX_REG;i++)release_reg(g,&g->reg[i]);dma_free_coherent(g->dev,ACT_BYTES,g->act,g->act_dma);dma_free_coherent(g->dev,OUT_BYTES,g->out,g->out_dma);return 0;}static const struct of_device_id match[]={ {.compatible="openai,gemv-q2k-u96"},{}};MODULE_DEVICE_TABLE(of,match);static struct platform_driver drv={.probe=probe,.remove=remove_,.driver={.name=NAME,.of_match_table=match}};module_platform_driver(drv);MODULE_LICENSE("GPL");
