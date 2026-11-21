#include <linux/module.h>
#include <linux/fs.h>
#include <linux/io.h>
#include <linux/dma-mapping.h>
#include <linux/platform_device.h>
#include <linux/device.h>
#include <linux/uaccess.h>
#include <linux/mutex.h>
#include <linux/jiffies.h>
#include <linux/slab.h>
#include <linux/ktime.h>
#include "gemv_q2k_uapi.h"

#define GEMV_Q2K_BASE 0xA0000000UL
#define GEMV_Q2K_SPAN 0x1000
#define GEMV_Q2K_TILE_BYTES (4U * 1024U * 1024U)
#define GEMV_Q2K_ACT_BYTES  (1U * 1024U * 1024U)
#define GEMV_Q2K_OUT_BYTES  (64U * 1024U)

#define MAX_SLABS 24
#define MAX_CACHE_ENTRIES 1024

#define R32(x) readl(regs + (x))
#define W32(v,x) writel((v), regs + (x))

enum { CTRL=0, W0=0x10, Y0=0x20, O0=0x30, W1=0x40, Y1=0x50, O1=0x60, NB=0x70, R0=0x80, R1=0x90, STATUS=0xa0 };

static void __iomem *regs;
static DEFINE_MUTEX(lock);
static int major;
static struct platform_device *pdev;
static struct class *dev_cls;
static struct device *dev_node;

struct buf { void *cpu; dma_addr_t dma; size_t sz; };
static struct buf bw0, by0, bo0, bw1, by1, bo1;

struct slab_buf { void *cpu; dma_addr_t dma; size_t used; };
static struct slab_buf slabs[MAX_SLABS];
static int num_slabs;

struct wcache_entry {
    u64 user_addr;
    size_t len;
    dma_addr_t dma;
};
static struct wcache_entry *wcache;
static int num_entries;

static u64 stat_calls, stat_hits, stat_misses, stat_copy_ns, stat_hw_ns;

static int allocb(struct buf *b, size_t sz) {
    b->sz = sz;
    b->cpu = dma_alloc_coherent(&pdev->dev, sz, &b->dma, GFP_KERNEL);
    return b->cpu ? 0 : -ENOMEM;
}

static void freeb(struct buf *b) {
    if (b->cpu && pdev) {
        dma_free_coherent(&pdev->dev, b->sz, b->cpu, b->dma);
        b->cpu = NULL;
    }
}

static void ptr(unsigned r, dma_addr_t a) {
    W32(lower_32_bits(a), r);
    W32(upper_32_bits(a), r + 4);
}

static dma_addr_t get_weight_dma(u64 uaddr, size_t len, struct buf *fallback, int *err) {
    int i, s;
    size_t aligned_len = ALIGN(len, 64);

    for (i = 0; i < num_entries; i++) {
        if (wcache[i].user_addr == uaddr && wcache[i].len == len) {
            stat_hits++;
            return wcache[i].dma;
        }
    }

    stat_misses++;
    if (num_entries < MAX_CACHE_ENTRIES && aligned_len <= GEMV_Q2K_TILE_BYTES) {
        for (s = 0; s < num_slabs; s++) {
            if (slabs[s].used + aligned_len <= GEMV_Q2K_TILE_BYTES) {
                void *dst = (u8 *)slabs[s].cpu + slabs[s].used;
                dma_addr_t dma = slabs[s].dma + slabs[s].used;
                ktime_t t0 = ktime_get();
                if (copy_from_user(dst, (void __user *)(unsigned long)uaddr, len)) {
                    *err = -EFAULT;
                    return 0;
                }
                stat_copy_ns += ktime_to_ns(ktime_sub(ktime_get(), t0));
                slabs[s].used += aligned_len;
                wcache[num_entries].user_addr = uaddr;
                wcache[num_entries].len = len;
                wcache[num_entries].dma = dma;
                num_entries++;
                return dma;
            }
        }
    }

    {
        ktime_t t0 = ktime_get();
        if (copy_from_user(fallback->cpu, (void __user *)(unsigned long)uaddr, len)) {
            *err = -EFAULT;
            return 0;
        }
        stat_copy_ns += ktime_to_ns(ktime_sub(ktime_get(), t0));
    }
    return fallback->dma;
}

static long submit(struct gemv_q2k_job __user *u) {
    struct gemv_q2k_job j;
    size_t wb, yb, ob0, ob1, wlen0, wlen1;
    dma_addr_t dw0 = 0, dw1 = 0;
    unsigned long end;
    ktime_t t0, t1;
    int rc = 0;

    if (copy_from_user(&j, u, sizeof(j))) return -EFAULT;
    if (!j.nb || j.nb > 1024 || j.rows0 + j.rows1 > 4096) return -EINVAL;

    wb = (size_t)j.nb * 84;
    yb = (size_t)j.nb * 8 * 34;
    wlen0 = wb * j.rows0;
    wlen1 = wb * j.rows1;
    ob0 = (size_t)j.rows0 * 4;
    ob1 = (size_t)j.rows1 * 4;

    if (wlen0 > GEMV_Q2K_TILE_BYTES || wlen1 > GEMV_Q2K_TILE_BYTES ||
        yb > GEMV_Q2K_ACT_BYTES || ob0 > GEMV_Q2K_OUT_BYTES || ob1 > GEMV_Q2K_OUT_BYTES)
        return -E2BIG;

    mutex_lock(&lock);
    stat_calls++;

    if (wlen0) {
        dw0 = get_weight_dma(j.weights, wlen0, &bw0, &rc);
        if (rc) goto out;
    }
    if (wlen1) {
        dw1 = get_weight_dma(j.weights + wlen0, wlen1, &bw1, &rc);
        if (rc) goto out;
    }

    t0 = ktime_get();
    if (copy_from_user(by0.cpu, (void __user *)(unsigned long)j.activation, yb)) {
        rc = -EFAULT;
        goto out;
    }
    memcpy(by1.cpu, by0.cpu, yb);
    t1 = ktime_get();
    stat_copy_ns += ktime_to_ns(ktime_sub(t1, t0));

    ptr(W0, dw0); ptr(Y0, by0.dma); ptr(O0, bo0.dma);
    ptr(W1, dw1); ptr(Y1, by1.dma); ptr(O1, bo1.dma);
    W32(j.nb, NB);
    W32(j.rows0, R0);
    W32(j.rows1, R1);
    W32(1, CTRL);

    end = jiffies + msecs_to_jiffies(j.timeout_ms ?: 1000);
    while (!(R32(CTRL) & 2)) {
        if (time_after(jiffies, end)) {
            rc = -ETIMEDOUT;
            goto out;
        }
    }
    stat_hw_ns += ktime_to_ns(ktime_sub(ktime_get(), t1));

    j.status = R32(STATUS);
    if (j.status & 4) {
        rc = -EIO;
        goto out;
    }

    if ((ob0 && copy_to_user((void __user *)(unsigned long)j.output, bo0.cpu, ob0)) ||
        (ob1 && copy_to_user((void __user *)(unsigned long)(j.output + ob0), bo1.cpu, ob1)))
        rc = -EFAULT;

out:
    mutex_unlock(&lock);
    if (!rc && copy_to_user(u, &j, sizeof(j))) rc = -EFAULT;
    return rc;
}

static long ioctl_(struct file *f, unsigned c, unsigned long a) {
    return c == GEMV_Q2K_SUBMIT ? submit((void __user *)a) : -ENOTTY;
}
static const struct file_operations fops = { .owner = THIS_MODULE, .unlocked_ioctl = ioctl_ };

static void cleanup_all(void) {
    int i;
    if (dev_node && !IS_ERR(dev_node)) device_destroy(dev_cls, MKDEV(major, 0));
    if (dev_cls && !IS_ERR(dev_cls)) class_destroy(dev_cls);
    if (major > 0) unregister_chrdev(major, "gemv_q2k");
    if (regs) iounmap(regs);
    for (i = 0; i < num_slabs; i++) {
        if (slabs[i].cpu && pdev)
            dma_free_coherent(&pdev->dev, GEMV_Q2K_TILE_BYTES, slabs[i].cpu, slabs[i].dma);
    }
    kfree(wcache);
    freeb(&bw0); freeb(&by0); freeb(&bo0); freeb(&bw1); freeb(&by1); freeb(&bo1);
    if (pdev && !IS_ERR(pdev)) platform_device_unregister(pdev);
}

static int __init init_(void) {
    int r, i;
    pdev = platform_device_register_simple("gemv_q2k", -1, NULL, 0);
    if (IS_ERR(pdev)) return PTR_ERR(pdev);

    r = dma_set_mask_and_coherent(&pdev->dev, DMA_BIT_MASK(32));
    if (r) { platform_device_unregister(pdev); return r; }

    if ((r = allocb(&bw0, GEMV_Q2K_TILE_BYTES)) ||
        (r = allocb(&bw1, GEMV_Q2K_TILE_BYTES)) ||
        (r = allocb(&by0, GEMV_Q2K_ACT_BYTES))  ||
        (r = allocb(&by1, GEMV_Q2K_ACT_BYTES))  ||
        (r = allocb(&bo0, GEMV_Q2K_OUT_BYTES))  ||
        (r = allocb(&bo1, GEMV_Q2K_OUT_BYTES))) {
        cleanup_all();
        return r;
    }

    wcache = kcalloc(MAX_CACHE_ENTRIES, sizeof(*wcache), GFP_KERNEL);
    if (!wcache) { cleanup_all(); return -ENOMEM; }

    for (i = 0; i < MAX_SLABS; i++) {
        slabs[i].cpu = dma_alloc_coherent(&pdev->dev, GEMV_Q2K_TILE_BYTES, &slabs[i].dma, GFP_KERNEL);
        if (!slabs[i].cpu) break;
        slabs[i].used = 0;
        num_slabs++;
    }
    pr_info("gemv_q2k: allocated %d CMA slabs (%d MB weight cache)\n", num_slabs, num_slabs * 4);

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

static void __exit exit_(void) {
    pr_info("gemv_q2k: calls=%llu hits=%llu misses=%llu copy_ms=%llu hw_ms=%llu cached=%d\n",
            stat_calls, stat_hits, stat_misses,
            stat_copy_ns / 1000000ULL, stat_hw_ns / 1000000ULL, num_entries);
    cleanup_all();
}

module_init(init_);
module_exit(exit_);
MODULE_LICENSE("GPL");
