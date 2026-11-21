// Platform-specific constants below must match the Vivado address editor.
#include <linux/module.h>
#include <linux/fs.h>
#include <linux/io.h>
#include <linux/dma-mapping.h>
#include <linux/platform_device.h>
#include <linux/device.h>
#include <linux/uaccess.h>
#include <linux/mutex.h>
#include <linux/jiffies.h>
#include "gemv_q2k_uapi.h"

#define GEMV_Q2K_BASE 0xA0000000UL /* CHANGE: AXI-Lite base */
#define GEMV_Q2K_SPAN 0x1000
#define GEMV_Q2K_TILE_BYTES (4U * 1024U * 1024U) /* CHANGE to available CMA */

#define R32(x) readl(regs + (x))
#define W32(v,x) writel((v), regs + (x))

enum { CTRL=0, W0=0x10,Y0=0x20,O0=0x30,W1=0x40,Y1=0x50,O1=0x60,NB=0x70,R0=0x80,R1=0x90,STATUS=0xa0 };

static void __iomem *regs;
static DEFINE_MUTEX(lock);
static int major;
static struct platform_device *pdev;
static struct class *dev_cls;
static struct device *dev_node;

struct buf { void *cpu; dma_addr_t dma; };
static struct buf bw0,by0,bo0,bw1,by1,bo1;

static int allocb(struct buf *b){
    b->cpu = dma_alloc_coherent(&pdev->dev, GEMV_Q2K_TILE_BYTES, &b->dma, GFP_KERNEL);
    return b->cpu ? 0 : -ENOMEM;
}
static void freeb(struct buf *b){
    if (b->cpu && pdev) {
        dma_free_coherent(&pdev->dev, GEMV_Q2K_TILE_BYTES, b->cpu, b->dma);
        b->cpu = NULL;
    }
}
static void ptr(unsigned r,dma_addr_t a){W32(lower_32_bits(a),r);W32(upper_32_bits(a),r+4);}

static long submit(struct gemv_q2k_job __user *u) {
 struct gemv_q2k_job j; size_t wb,yb,ob; unsigned long end; int rc=0;
 if(copy_from_user(&j,u,sizeof(j))) return -EFAULT;
 if(!j.nb||j.nb>1024||j.rows0+j.rows1>4096) return -EINVAL;
 wb=(size_t)j.nb*84; yb=(size_t)j.nb*8*34; ob=(size_t)(j.rows0+j.rows1)*4;
 if(wb*j.rows0>GEMV_Q2K_TILE_BYTES||wb*j.rows1>GEMV_Q2K_TILE_BYTES||yb>GEMV_Q2K_TILE_BYTES||ob>GEMV_Q2K_TILE_BYTES)return -E2BIG;
 mutex_lock(&lock);
 if(copy_from_user(bw0.cpu,(void __user *)(unsigned long)j.weights,wb*j.rows0)||
    copy_from_user(bw1.cpu,(void __user *)(unsigned long)(j.weights+wb*j.rows0),wb*j.rows1)||
    copy_from_user(by0.cpu,(void __user *)(unsigned long)j.activation,yb)){rc=-EFAULT;goto out;}
 memcpy(by1.cpu,by0.cpu,yb); ptr(W0,bw0.dma);ptr(Y0,by0.dma);ptr(O0,bo0.dma);ptr(W1,bw1.dma);ptr(Y1,by1.dma);ptr(O1,bo1.dma);
 W32(j.nb,NB);W32(j.rows0,R0);W32(j.rows1,R1);W32(1,CTRL); end=jiffies+msecs_to_jiffies(j.timeout_ms?:1000);
 while(!(R32(CTRL)&2))if(time_after(jiffies,end)){rc=-ETIMEDOUT;goto out;}
 j.status=R32(STATUS); if(j.status&4){rc=-EIO;goto out;}
 if(copy_to_user((void __user *)(unsigned long)j.output,bo0.cpu,j.rows0*4)||copy_to_user((void __user *)(unsigned long)(j.output+j.rows0*4),bo1.cpu,j.rows1*4))rc=-EFAULT;
out: mutex_unlock(&lock); if(!rc&&copy_to_user(u,&j,sizeof(j)))rc=-EFAULT; return rc;
}

static long ioctl_(struct file *f,unsigned c,unsigned long a){return c==GEMV_Q2K_SUBMIT?submit((void __user *)a):-ENOTTY;}
static const struct file_operations fops={.owner=THIS_MODULE,.unlocked_ioctl=ioctl_};

static void cleanup_all(void){
    if (dev_node && !IS_ERR(dev_node)) device_destroy(dev_cls, MKDEV(major, 0));
    if (dev_cls && !IS_ERR(dev_cls)) class_destroy(dev_cls);
    if (major > 0) unregister_chrdev(major, "gemv_q2k");
    if (regs) iounmap(regs);
    freeb(&bw0); freeb(&by0); freeb(&bo0); freeb(&bw1); freeb(&by1); freeb(&bo1);
    if (pdev && !IS_ERR(pdev)) platform_device_unregister(pdev);
}

static int __init init_(void){
    int r;
    pdev = platform_device_register_simple("gemv_q2k", -1, NULL, 0);
    if (IS_ERR(pdev)) return PTR_ERR(pdev);

    r = dma_set_mask_and_coherent(&pdev->dev, DMA_BIT_MASK(32));
    if (r) { platform_device_unregister(pdev); return r; }

    if ((r=allocb(&bw0))||(r=allocb(&by0))||(r=allocb(&bo0))||(r=allocb(&bw1))||(r=allocb(&by1))||(r=allocb(&bo1))){
        cleanup_all();
        return r;
    }
    regs = ioremap(GEMV_Q2K_BASE, GEMV_Q2K_SPAN);
    if (!regs) { cleanup_all(); return -ENOMEM; }

    major = register_chrdev(0, "gemv_q2k", &fops);
    if (major < 0) { cleanup_all(); return major; }

    dev_cls = class_create(THIS_MODULE, "gemv_q2k");
    if (IS_ERR(dev_cls)) { r = PTR_ERR(dev_cls); cleanup_all(); return r; }

    dev_node = device_create(dev_cls, NULL, MKDEV(major, 0), NULL, "gemv_q2k");
    if (IS_ERR(dev_node)) { r = PTR_ERR(dev_node); cleanup_all(); return r; }

    return 0;
}

static void __exit exit_(void){ cleanup_all(); }
module_init(init_);module_exit(exit_);MODULE_LICENSE("GPL");