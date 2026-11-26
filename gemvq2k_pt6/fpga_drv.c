// Ultra96-V2 Q2_K driver: pin model pages once; no token-time weight copies.
#include <linux/module.h>
#include <linux/platform_device.h>
#include <linux/of.h>
#include <linux/miscdevice.h>
#include <linux/fs.h>
#include <linux/io.h>
#include <linux/iopoll.h>
#include <linux/dma-mapping.h>
#include <linux/mm.h>
#include <linux/uaccess.h>
#include <linux/mutex.h>
#include <linux/jiffies.h>
#include <linux/slab.h>
#include <linux/of_address.h>
#include "gemv_q2k_uapi.h"
#define NAME "gemv_q2k"
#define MAX_REG 256
#define MAX_BLOCKS 64
#define MAX_ROWS 8192
#define ACT_BYTES (MAX_BLOCKS*292)
#define OUT_BYTES (MAX_ROWS*4)
#define MAX_COPIED_PAGES ((160UL * 1024 * 1024) / PAGE_SIZE)
enum { CTRL=0, PT=0x10, ACT=0x20, OUT=0x30, WOFF=0x40, START=0x50, NB=0x60,
       ROWS=0x70, STATUS=0x80, WEIGHT_MEMORY=0x90 };
struct regn {
    u64 bytes;
    u32 page_offset;
    unsigned npages, pinned, copied, mapped;
    struct page **pages;
    u64 *pt_cpu;
    dma_addr_t pt_dma;
};
struct gemv { struct device *dev; void __iomem *r; struct miscdevice misc; struct mutex lock; struct regn reg[MAX_REG]; void *act,*out; dma_addr_t act_dma,out_dma; unsigned copied_pages; u32 capabilities; };
static inline u32 rd(struct gemv*g,u32 o){return readl(g->r+o);} static inline void wr(struct gemv*g,u32 o,u32 v){writel(v,g->r+o);} static void ptr(struct gemv*g,u32 o,dma_addr_t a){wr(g,o,lower_32_bits(a));wr(g,o+4,upper_32_bits(a));}
static void release_reg(struct gemv *g, struct regn *x)
{
    unsigned i;

    for (i = 0; i < x->mapped; ++i)
        dma_unmap_page(g->dev, x->pt_cpu[i], PAGE_SIZE, DMA_TO_DEVICE);
    if (x->pinned)
        unpin_user_pages(x->pages, x->pinned);
    for (i = 0; i < x->copied; ++i)
        __free_page(x->pages[i]);
    g->copied_pages -= x->copied;
    if (x->pt_cpu)
        dma_free_coherent(g->dev, x->npages * sizeof(*x->pt_cpu),
                          x->pt_cpu, x->pt_dma);
    kvfree(x->pages);
    memset(x, 0, sizeof(*x));
}

/* Keep a bounded copy only when a valid user mapping cannot be pinned. */
static int copy_weight_pages(struct gemv *g, struct regn *x,
                             const char __user *source)
{
    size_t copied = 0;
    unsigned i;

    if (x->npages > MAX_COPIED_PAGES - g->copied_pages)
        return -ENOMEM;

    if (x->pinned) {
        unpin_user_pages(x->pages, x->pinned);
        x->pinned = 0;
    }

    for (i = 0; i < x->npages; ++i) {
        const unsigned offset = i == 0 ? x->page_offset : 0;
        const size_t count = min_t(size_t, PAGE_SIZE - offset, x->bytes - copied);
        struct page *page = alloc_page(GFP_KERNEL | __GFP_ZERO);

        if (!page)
            return -ENOMEM;
        x->pages[i] = page;
        ++x->copied;
        ++g->copied_pages;
        if (copy_from_user((char *)page_address(page) + offset, source + copied, count))
            return -EFAULT;
        copied += count;
    }

    return 0;
}

static long reg_weight(struct gemv *g, struct gemv_q2k_register __user *u)
{
    struct gemv_q2k_register q;
    struct regn *x;
    unsigned i, slot;
    unsigned long first;
    int got, rc = -ENOMEM;

    if (copy_from_user(&q, u, sizeof(q)))
        return -EFAULT;
    if (!q.bytes || q.bytes > (u64)MAX_BLOCKS * MAX_ROWS * 110 ||
        !access_ok(u64_to_user_ptr(q.address), q.bytes))
        return -EINVAL;

    mutex_lock(&g->lock);
    for (slot = 0; slot < MAX_REG && g->reg[slot].pages; ++slot)
        ;
    if (slot == MAX_REG) {
        rc = -ENOSPC;
        goto unlock;
    }

    x = &g->reg[slot];
    first = (unsigned long)q.address & PAGE_MASK;
    x->page_offset = offset_in_page(q.address);
    x->bytes = q.bytes;
    x->npages = DIV_ROUND_UP(x->page_offset + q.bytes, PAGE_SIZE);
    x->pages = kvmalloc_array(x->npages, sizeof(*x->pages), GFP_KERNEL);
    if (!x->pages)
        goto fail;
    x->pt_cpu = dma_alloc_coherent(g->dev, x->npages * sizeof(*x->pt_cpu),
                                   &x->pt_dma, GFP_KERNEL);
    if (!x->pt_cpu)
        goto fail;

    while (x->pinned < x->npages) {
        const unsigned count = min(x->npages - x->pinned, 1024u);

        got = pin_user_pages_fast(first + (unsigned long)x->pinned * PAGE_SIZE,
                                  count, FOLL_LONGTERM, x->pages + x->pinned);
        if (got <= 0) {
            rc = got < 0 ? got : -EFAULT;
            if (rc == -ENOMEM)
                rc = copy_weight_pages(g, x, u64_to_user_ptr(q.address));
            if (rc) {
                dev_warn_ratelimited(g->dev,
                                     "weight registration failed: %d (copied %u/%lu pages)\n",
                                     rc, g->copied_pages, MAX_COPIED_PAGES);
                goto fail;
            }
            dev_dbg(g->dev, "copied %u weight pages after pin failure\n", x->copied);
            break;
        }
        x->pinned += got;
        cond_resched();
    }
    for (i = 0; i < x->npages; ++i) {
        dma_addr_t address = dma_map_page(g->dev, x->pages[i], 0,
                                          PAGE_SIZE, DMA_TO_DEVICE);

        if (dma_mapping_error(g->dev, address)) {
            rc = -ENOMEM;
            goto fail;
        }
        x->pt_cpu[i] = address;
        ++x->mapped;
    }

    q.handle = slot + 1;
    if (copy_to_user(u, &q, sizeof(q))) {
        rc = -EFAULT;
        goto fail;
    }
    rc = 0;
    goto unlock;
fail:
    release_reg(g, x);
unlock:
    mutex_unlock(&g->lock);
    return rc;
}
static long submit(struct gemv *g, struct gemv_q2k_submit __user *u)
{
    struct gemv_q2k_submit q;
    struct regn *weights;
    unsigned long deadline;
    size_t activation_bytes, output_bytes;
    u32 start, block_bytes;
    int rc = 0;

    if (copy_from_user(&q, u, sizeof(q)))
        return -EFAULT;
    if (!q.handle || q.handle > MAX_REG || !q.nb || q.nb > MAX_BLOCKS ||
        !q.rows || q.rows > MAX_ROWS)
        return -EINVAL;

    block_bytes = 84;
    start = q.start_block;
    if (start & GEMV_NATIVE_Q3_START) {
        if (!(g->capabilities & GEMV_CAP_NATIVE_Q3))
            return -EOPNOTSUPP;
        block_bytes = 110;
        start &= ~GEMV_NATIVE_Q3_START;
    }
    activation_bytes = (size_t)q.nb * 292;
    output_bytes = (size_t)q.rows * sizeof(u32);
    mutex_lock(&g->lock);
    weights = &g->reg[q.handle - 1];
    if (!weights->pages || ((u64)start + (u64)q.rows * q.nb) * block_bytes > weights->bytes) {
        rc = -EINVAL;
        goto out;
    }
    if (copy_from_user(g->act, u64_to_user_ptr(q.activation), activation_bytes)) {
        rc = -EFAULT;
        goto out;
    }
    /* Page-table entries already contain absolute DMA byte addresses. */
    ptr(g, WEIGHT_MEMORY, 0);
    ptr(g, PT, weights->pt_dma);
    ptr(g, ACT, g->act_dma);
    ptr(g, OUT, g->out_dma);
    wr(g, WOFF, weights->page_offset);
    wr(g, START, q.start_block);
    wr(g, NB, q.nb);
    wr(g, ROWS, q.rows);
    wr(g, CTRL, 1);
    deadline = jiffies + msecs_to_jiffies(q.timeout_ms ?: 1000);
    while (!(rd(g, CTRL) & 2)) {
        if (time_after(jiffies, deadline)) {
            rc = -ETIMEDOUT;
            goto out;
        }
    }
    q.status = rd(g, STATUS);
    if (q.status != 2) {
        rc = -EIO;
        goto out;
    }
    if (copy_to_user(u64_to_user_ptr(q.output), g->out, output_bytes))
        rc = -EFAULT;
out:
    mutex_unlock(&g->lock);
    return !rc && copy_to_user(u, &q, sizeof(q)) ? -EFAULT : rc;
}

static long ioctl_(struct file *f, unsigned int command, unsigned long argument)
{
    struct gemv *g = container_of(f->private_data, struct gemv, misc);
    void __user *arg = (void __user *)argument;

    switch (command) {
    case GEMV_Q2K_REGISTER:
        return reg_weight(g, arg);
    case GEMV_Q2K_SUBMIT:
        return submit(g, arg);
    case GEMV_Q2K_CAPABILITIES:
        return put_user(g->capabilities, (u32 __user *)arg);
    default:
        return -ENOTTY;
    }
}

static const struct file_operations fops = {
    .owner = THIS_MODULE, .unlocked_ioctl = ioctl_,
};
static int probe(struct platform_device *p)
{
    struct resource *resource = platform_get_resource(p, IORESOURCE_MEM, 0);
    struct gemv *g;
    int rc;

    if (PAGE_SIZE != 4096 || !resource || resource_size(resource) < 0x98)
        return -EINVAL;
    /* A PYNQ generic-uio node requires an explicit driver_override. */
    g = devm_kzalloc(&p->dev, sizeof(*g), GFP_KERNEL);
    if (!g)
        return -ENOMEM;
    g->dev = &p->dev;
    mutex_init(&g->lock);
    g->r = devm_platform_ioremap_resource(p, 0);
    if (IS_ERR(g->r))
        return PTR_ERR(g->r);
    if (of_device_is_compatible(p->dev.of_node, "openai,gemv-q2q3-u96")) {
        u32 ctrl;

        rc = readl_poll_timeout(g->r + CTRL, ctrl, ctrl & 4, 1, 10000);
        if (rc)
            return rc;
        /* This signature transaction has no DMA and is invalid on old IP. */
        wr(g, START, 0xffffffffu);
        wr(g, NB, 0);
        wr(g, ROWS, 0);
        wr(g, CTRL, 1);
        rc = readl_poll_timeout(g->r + CTRL, ctrl, ctrl & 2, 1, 10000);
        if (rc || rd(g, STATUS) != 0x51334b31u)
            return dev_err_probe(g->dev, -ENODEV, "native Q3_K FPGA signature missing\n");
        g->capabilities = GEMV_CAP_NATIVE_Q3;
    }
    if (!p->dev.dma_mask)
        p->dev.dma_mask = &p->dev.coherent_dma_mask;
    rc = dma_set_mask_and_coherent(g->dev, DMA_BIT_MASK(31));
    if (rc)
        return rc;

    g->act = dmam_alloc_coherent(g->dev, ACT_BYTES, &g->act_dma, GFP_KERNEL);
    g->out = dmam_alloc_coherent(g->dev, OUT_BYTES, &g->out_dma, GFP_KERNEL);
    if (!g->act || !g->out)
        return -ENOMEM;
    g->misc = (struct miscdevice) {
        .minor = MISC_DYNAMIC_MINOR, .name = NAME, .fops = &fops,
        .parent = &p->dev,
    };
    platform_set_drvdata(p, g);
    return misc_register(&g->misc);
}

static int remove_(struct platform_device *p)
{
    struct gemv *g = platform_get_drvdata(p);
    unsigned i;

    misc_deregister(&g->misc);
    for (i = 0; i < MAX_REG; i++)
        release_reg(g, &g->reg[i]);
    return 0;
}

static const struct of_device_id match[] = {
    { .compatible = "openai,gemv-q2q3-u96" },
    { .compatible = "openai,gemv-q2k-u96" },
    { }
};
MODULE_DEVICE_TABLE(of, match);
static struct platform_driver drv = {
    .probe = probe,
    .remove = remove_,
    .driver = { .name = NAME, .of_match_table = match },
};
module_platform_driver(drv);
MODULE_LICENSE("GPL");
