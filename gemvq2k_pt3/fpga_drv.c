// SPDX-License-Identifier: GPL-2.0
/*
 * gemv_q2k: driver for the 2-engine K-quant decode-GEMV accelerator.
 *
 * Zero-copy weights: REGISTER pins llama.cpp's pages (pin_user_pages_fast,
 * FOLL_LONGTERM), cleans them from the CPU caches once (dma_map_page) and
 * writes their bus addresses into a page table the FPGA walks. Weights never
 * get copied and need no CMA; only the page tables (8 B per 4 KiB page,
 * ~1.3 MB for a 1.5B Q2_K model), the activation (70 KB) and the output
 * buffer (1 MB) are DMA allocations, which fit the default 128 MB CMA.
 *
 * Binding: device-tree node "xlnx,gemv-q2k-accel-1.0" if present, else a
 * platform device at module parameter `base` (Vivado address editor).
 */
#include <linux/module.h>
#include <linux/platform_device.h>
#include <linux/of.h>
#include <linux/miscdevice.h>
#include <linux/fs.h>
#include <linux/io.h>
#include <linux/mm.h>
#include <linux/dma-mapping.h>
#include <linux/uaccess.h>
#include <linux/mutex.h>
#include <linux/idr.h>
#include <linux/slab.h>
#include <linux/jiffies.h>
#include <linux/sched.h>
#include <linux/version.h>
#include "gemv_q2k_uapi.h"

#define DRV_NAME "gemv_q2k"

static unsigned long base = 0xA0000000UL;      /* M_AXI_HPM0_FPD default */
module_param(base, ulong, 0444);
MODULE_PARM_DESC(base, "AXI-Lite base address when no device-tree node exists");
static unsigned int max_rows = 262144;
module_param(max_rows, uint, 0444);
MODULE_PARM_DESC(max_rows, "largest matrix row count accepted (output buffer size)");
static unsigned int max_pin_mb = 1024;
module_param(max_pin_mb, uint, 0444);
MODULE_PARM_DESC(max_pin_mb, "largest single tensor registration in MiB");

/* AXI-Lite register map: must match hls/gemv_q2k_accel.h */
#define REG_CTRL      0x000
#define REG_M(i)      (0x010 + 0x10 * (i))
#define REG_O(e)      (0x050 + 0x10 * (e))
#define REG_TYPE      0x070
#define REG_NB        0x080
#define REG_YADDR     0x090
#define REG_PT        0x0a0
#define REG_WOFF(e)   (0x0b0 + 0x10 * (e))
#define REG_ROWS(e)   (0x0d0 + 0x10 * (e))
#define REG_OADDR(e)  (0x0f0 + 0x10 * (e))
#define REG_STATUS    0x110
#define REG_SPAN      0x1000
#define NSHADOW       (0x120 / 4)
#define AP_START      BIT(0)
#define AP_DONE       BIT(1)     /* clear-on-read */
#define AP_IDLE       BIT(2)
#define ST_DONE       2
#define ST_ERROR      4
#define HW_PAGE       4096u
#define PT_CHUNK      64u        /* the kernel reads page-table entries 64 at a time */
#define TYPE_MASK     (BIT(10) | BIT(11) | BIT(12) | BIT(13) | BIT(14))

#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 10, 0)
#define GEMV_NONCOHERENT 1
#endif

struct gemv_dev {
	struct device *dev;
	void __iomem *regs;
	struct miscdevice misc;
	struct mutex hw_lock;
	bool broken;
	void *act_cpu, *out_cpu;
	dma_addr_t act_dma, out_dma;
	size_t act_bytes, out_bytes;
	u32 shadow[NSHADOW];
	bool shadow_ok[NSHADOW];
};

struct gemv_wbuf {               /* one registered (pinned) tensor */
	struct page **pages;
	unsigned long npages;
	unsigned long mapped;
	u64 *pt_cpu;
	dma_addr_t pt_dma;
	size_t pt_bytes;
	u32 page_off;
	u64 bytes;
};

struct gemv_file {
	struct gemv_dev *g;
	struct mutex lock;
	struct idr bufs;
};

static u32 block_bytes(u32 t)
{
	switch (t) {
	case GEMV_Q2K_TYPE_Q2_K: return 84;
	case GEMV_Q2K_TYPE_Q3_K: return 110;
	case GEMV_Q2K_TYPE_Q4_K: return 144;
	case GEMV_Q2K_TYPE_Q5_K: return 176;
	case GEMV_Q2K_TYPE_Q6_K: return 210;
	}
	return 0;
}

/* ---- small DMA buffers (activation, output) ----------------------------- */
static void *g_alloc(struct device *d, size_t sz, dma_addr_t *h, enum dma_data_direction dir)
{
#ifdef GEMV_NONCOHERENT
	return dma_alloc_noncoherent(d, sz, h, dir, GFP_KERNEL | __GFP_NOWARN);
#else
	return dma_alloc_coherent(d, sz, h, GFP_KERNEL | __GFP_NOWARN);
#endif
}

static void g_free(struct device *d, size_t sz, void *cpu, dma_addr_t h, enum dma_data_direction dir)
{
#ifdef GEMV_NONCOHERENT
	dma_free_noncoherent(d, sz, cpu, h, dir);
#else
	dma_free_coherent(d, sz, cpu, h);
#endif
}

static void g_sync_dev(struct device *d, dma_addr_t h, size_t sz, enum dma_data_direction dir)
{
#ifdef GEMV_NONCOHERENT
	dma_sync_single_for_device(d, h, sz, dir);
#endif
}

static void g_sync_cpu(struct device *d, dma_addr_t h, size_t sz, enum dma_data_direction dir)
{
#ifdef GEMV_NONCOHERENT
	dma_sync_single_for_cpu(d, h, sz, dir);
#endif
}

/* ---- register access ---------------------------------------------------- */
static void wr32(struct gemv_dev *g, u32 off, u32 v)
{
	unsigned int i = off / 4;

	if (g->shadow_ok[i] && g->shadow[i] == v)
		return;
	writel(v, g->regs + off);
	g->shadow[i] = v;
	g->shadow_ok[i] = true;
}

static void wr64(struct gemv_dev *g, u32 off, u64 v)
{
	wr32(g, off, lower_32_bits(v));
	wr32(g, off + 4, upper_32_bits(v));
}

/* ---- pinned weight buffers ---------------------------------------------- */
static void wbuf_release(struct gemv_dev *g, struct gemv_wbuf *wb)
{
	unsigned long i;

	for (i = 0; i < wb->mapped; i++)
		dma_unmap_page(g->dev, (dma_addr_t)wb->pt_cpu[i], PAGE_SIZE, DMA_TO_DEVICE);
	if (wb->pages && wb->npages)
		unpin_user_pages(wb->pages, wb->npages);
	if (wb->pt_cpu)
		dma_free_coherent(g->dev, wb->pt_bytes, wb->pt_cpu, wb->pt_dma);
	kvfree(wb->pages);
	kfree(wb);
}

static long do_register(struct gemv_file *gf, void __user *u)
{
	struct gemv_dev *g = gf->g;
	struct gemv_q2k_register r;
	struct gemv_wbuf *wb;
	unsigned long start, done = 0, i;
	long rc;
	int id;

	if (copy_from_user(&r, u, sizeof(r)))
		return -EFAULT;
	if (!r.bytes || r.bytes > ((u64)max_pin_mb << 20) || (r.addr & 15))
		return -EINVAL;
	wb = kzalloc(sizeof(*wb), GFP_KERNEL);
	if (!wb)
		return -ENOMEM;
	start = r.addr & PAGE_MASK;
	wb->page_off = r.addr - start;
	wb->bytes = r.bytes;
	wb->npages = DIV_ROUND_UP(wb->page_off + r.bytes, PAGE_SIZE);
	wb->pages = kvmalloc_array(wb->npages, sizeof(*wb->pages), GFP_KERNEL);
	wb->pt_bytes = round_up(wb->npages, PT_CHUNK) * sizeof(u64);
	wb->pt_cpu = dma_alloc_coherent(g->dev, wb->pt_bytes, &wb->pt_dma, GFP_KERNEL | __GFP_NOWARN);
	if (!wb->pages || !wb->pt_cpu) {
		wb->npages = 0;
		rc = -ENOMEM;
		goto err;
	}
	memset(wb->pt_cpu, 0, wb->pt_bytes);

	/* Read-only long-term pin; works for llama.cpp's mmap and --no-mmap buffers. */
	while (done < wb->npages) {
		int n = (int)min_t(unsigned long, wb->npages - done, 16384);
		int got = pin_user_pages_fast(start + done * PAGE_SIZE, n, FOLL_LONGTERM, wb->pages + done);

		if (got <= 0) {
			unsigned long pinned = done;

			wb->npages = 0;                        /* unpin only what we hold */
			if (pinned)
				unpin_user_pages(wb->pages, pinned);
			rc = got ? got : -EFAULT;
			goto err;
		}
		done += got;
		cond_resched();
	}
	for (i = 0; i < wb->npages; i++) {
		dma_addr_t a = dma_map_page(g->dev, wb->pages[i], 0, PAGE_SIZE, DMA_TO_DEVICE);

		if (dma_mapping_error(g->dev, a)) {
			rc = -ENOMEM;
			goto err;
		}
		wb->pt_cpu[i] = a;
		wb->mapped = i + 1;
		if ((i & 1023) == 1023)
			cond_resched();
	}
	wmb();

	mutex_lock(&gf->lock);
	id = idr_alloc(&gf->bufs, wb, 1, INT_MAX, GFP_KERNEL);
	mutex_unlock(&gf->lock);
	if (id < 0) {
		rc = id;
		goto err;
	}
	r.handle = id;
	if (copy_to_user(u, &r, sizeof(r))) {
		mutex_lock(&gf->lock);
		idr_remove(&gf->bufs, id);
		mutex_unlock(&gf->lock);
		rc = -EFAULT;
		goto err;
	}
	return 0;
err:
	wbuf_release(g, wb);
	return rc;
}

static long do_free(struct gemv_file *gf, void __user *u)
{
	struct gemv_wbuf *wb;
	__u32 h;

	if (get_user(h, (__u32 __user *)u))
		return -EFAULT;
	mutex_lock(&gf->lock);              /* excludes a running SUBMIT on this file */
	wb = idr_remove(&gf->bufs, h);
	mutex_unlock(&gf->lock);
	if (!wb)
		return -ENOENT;
	wbuf_release(gf->g, wb);
	return 0;
}

static long do_info(struct gemv_file *gf, void __user *u)
{
	struct gemv_q2k_info i = {
		.version = GEMV_Q2K_ABI_VERSION,
		.engines = GEMV_Q2K_ENGINES,
		.max_blocks = GEMV_Q2K_MAX_BLOCKS,
		.act_block_bytes = GEMV_Q2K_ACT_BLOCK_BYTES,
		.max_rows = max_rows,
		.max_pin_bytes = (u64)max_pin_mb << 20,
		.type_mask = TYPE_MASK,
		.page_bytes = HW_PAGE,
	};

	return copy_to_user(u, &i, sizeof(i)) ? -EFAULT : 0;
}

static long do_submit(struct gemv_file *gf, void __user *u)
{
	struct gemv_dev *g = gf->g;
	struct gemv_q2k_job j;
	struct gemv_wbuf *wb;
	unsigned long deadline;
	size_t abytes, obytes;
	u32 B, chunk, n0, n1, spins = 0;
	long rc = 0;

	if (copy_from_user(&j, u, sizeof(j)))
		return -EFAULT;
	B = block_bytes(j.type);
	if (!B || !j.nb || j.nb > GEMV_Q2K_MAX_BLOCKS || !j.rows || j.rows > max_rows)
		return -EINVAL;
	abytes = (size_t)j.nb * GEMV_Q2K_ACT_BLOCK_BYTES;
	obytes = round_up((size_t)j.rows * 4, 16);

	mutex_lock(&gf->lock);
	wb = idr_find(&gf->bufs, j.handle);
	if (!wb) {
		rc = -ENOENT;
		goto out_file;
	}
	if ((u64)j.rows * j.nb * B > wb->bytes) {
		rc = -EINVAL;
		goto out_file;
	}

	mutex_lock(&g->hw_lock);
	if (g->broken) {
		rc = -EIO;
		goto out_hw;
	}
	if (copy_from_user(g->act_cpu, u64_to_user_ptr(j.activation), abytes)) {
		rc = -EFAULT;
		goto out_hw;
	}
	g_sync_dev(g->dev, g->act_dma, abytes, DMA_TO_DEVICE);
	g_sync_dev(g->dev, g->out_dma, obytes, DMA_FROM_DEVICE);

	/* Engine 0 takes a multiple of 8 rows: 8 rows of any K-quant are a
	 * multiple of 16 bytes, so engine 1 starts on a 128-bit boundary and its
	 * output (n0 * 4 bytes) is 32-byte aligned. */
	chunk = round_up(DIV_ROUND_UP(j.rows, 2), 8);
	n0 = min(chunk, j.rows);
	n1 = j.rows - n0;
	wr32(g, REG_TYPE, j.type);
	wr32(g, REG_NB, j.nb);
	wr64(g, REG_YADDR, g->act_dma);
	wr64(g, REG_PT, wb->pt_dma);
	wr32(g, REG_WOFF(0), wb->page_off);
	wr32(g, REG_WOFF(1), wb->page_off + (u32)((u64)n0 * j.nb * B));
	wr32(g, REG_ROWS(0), n0);
	wr32(g, REG_ROWS(1), n1);
	wr64(g, REG_OADDR(0), g->out_dma);
	wr64(g, REG_OADDR(1), g->out_dma + (u64)n0 * 4);

	if (!(readl(g->regs + REG_CTRL) & AP_IDLE)) {
		dev_err(g->dev, "accelerator not idle; disabling until reload\n");
		g->broken = true;
		rc = -EBUSY;
		goto out_hw;
	}
	writel(AP_START, g->regs + REG_CTRL);

	deadline = jiffies + msecs_to_jiffies(j.timeout_ms ? j.timeout_ms : 2000);
	while (!(readl(g->regs + REG_CTRL) & AP_DONE)) {
		if (time_after(jiffies, deadline)) {
			dev_err(g->dev, "job timeout (type=%u nb=%u rows=%u); disabling until reload\n",
				j.type, j.nb, j.rows);
			g->broken = true;
			rc = -ETIMEDOUT;
			goto out_hw;
		}
		if ((++spins & 1023) == 0)
			cond_resched();
		else
			cpu_relax();
	}
	j.status = readl(g->regs + REG_STATUS);
	if (!(j.status & ST_DONE) || (j.status & ST_ERROR)) {
		rc = -EIO;
		goto out_hw;
	}
	g_sync_cpu(g->dev, g->out_dma, obytes, DMA_FROM_DEVICE);
	if (copy_to_user(u64_to_user_ptr(j.output), g->out_cpu, (size_t)j.rows * 4))
		rc = -EFAULT;
out_hw:
	mutex_unlock(&g->hw_lock);
out_file:
	mutex_unlock(&gf->lock);
	if (!rc && copy_to_user(u, &j, sizeof(j)))
		rc = -EFAULT;
	return rc;
}

static long gemv_ioctl(struct file *f, unsigned int cmd, unsigned long arg)
{
	struct gemv_file *gf = f->private_data;
	void __user *u = (void __user *)arg;

	switch (cmd) {
	case GEMV_Q2K_INFO:     return do_info(gf, u);
	case GEMV_Q2K_REGISTER: return do_register(gf, u);
	case GEMV_Q2K_FREE:     return do_free(gf, u);
	case GEMV_Q2K_SUBMIT:   return do_submit(gf, u);
	}
	return -ENOTTY;
}

static int gemv_open(struct inode *ino, struct file *f)
{
	struct gemv_dev *g = container_of(f->private_data, struct gemv_dev, misc);
	struct gemv_file *gf = kzalloc(sizeof(*gf), GFP_KERNEL);

	if (!gf)
		return -ENOMEM;
	gf->g = g;
	mutex_init(&gf->lock);
	idr_init(&gf->bufs);
	f->private_data = gf;
	return 0;
}

static int free_one(int id, void *p, void *data)
{
	wbuf_release(data, p);
	return 0;
}

static int gemv_release(struct inode *ino, struct file *f)
{
	struct gemv_file *gf = f->private_data;

	idr_for_each(&gf->bufs, free_one, gf->g);
	idr_destroy(&gf->bufs);
	kfree(gf);
	return 0;
}

static const struct file_operations gemv_fops = {
	.owner = THIS_MODULE,
	.open = gemv_open,
	.release = gemv_release,
	.unlocked_ioctl = gemv_ioctl,
#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 5, 0)
	.compat_ioctl = compat_ptr_ioctl,
#endif
	.llseek = noop_llseek,
};

/* ---- platform glue ------------------------------------------------------ */
static int gemv_probe(struct platform_device *pdev)
{
	struct gemv_dev *g;
	int rc, i;

	if (PAGE_SIZE != HW_PAGE) {
		dev_err(&pdev->dev, "kernel PAGE_SIZE must be 4 KiB\n");
		return -EINVAL;
	}
	g = devm_kzalloc(&pdev->dev, sizeof(*g), GFP_KERNEL);
	if (!g)
		return -ENOMEM;
	g->dev = &pdev->dev;
	g->regs = devm_platform_ioremap_resource(pdev, 0);
	if (IS_ERR(g->regs))
		return PTR_ERR(g->regs);
	if (!pdev->dev.dma_mask)        /* self-created device on old kernels */
		pdev->dev.dma_mask = &pdev->dev.coherent_dma_mask;
	rc = dma_set_mask_and_coherent(g->dev, DMA_BIT_MASK(64));
	if (rc)
		rc = dma_set_mask_and_coherent(g->dev, DMA_BIT_MASK(32));
	if (rc)
		return rc;
	mutex_init(&g->hw_lock);

	g->act_bytes = PAGE_ALIGN(GEMV_Q2K_MAX_BLOCKS * GEMV_Q2K_ACT_BLOCK_BYTES);
	g->out_bytes = PAGE_ALIGN((size_t)max_rows * 4 + 64);
	g->act_cpu = g_alloc(g->dev, g->act_bytes, &g->act_dma, DMA_TO_DEVICE);
	g->out_cpu = g_alloc(g->dev, g->out_bytes, &g->out_dma, DMA_FROM_DEVICE);
	if (!g->act_cpu || !g->out_cpu) {
		rc = -ENOMEM;
		goto err;
	}
	for (i = 0; i < 4; i++)          /* "all of DDR" pointers: base 0 */
		wr64(g, REG_M(i), 0);
	for (i = 0; i < 2; i++)
		wr64(g, REG_O(i), 0);

	g->misc.minor = MISC_DYNAMIC_MINOR;
	g->misc.name = DRV_NAME;
	g->misc.fops = &gemv_fops;
	g->misc.parent = g->dev;
	/* User space never supplies DMA addresses, only its own mapped pages. */
	g->misc.mode = 0666;
	platform_set_drvdata(pdev, g);
	rc = misc_register(&g->misc);
	if (rc)
		goto err;
	dev_info(g->dev, "K-quant GEMV accelerator: %d engines, zero-copy weights, /dev/%s\n",
		 GEMV_Q2K_ENGINES, DRV_NAME);
	return 0;
err:
	if (g->act_cpu)
		g_free(g->dev, g->act_bytes, g->act_cpu, g->act_dma, DMA_TO_DEVICE);
	if (g->out_cpu)
		g_free(g->dev, g->out_bytes, g->out_cpu, g->out_dma, DMA_FROM_DEVICE);
	return rc;
}

static void gemv_remove_common(struct platform_device *pdev)
{
	struct gemv_dev *g = platform_get_drvdata(pdev);

	misc_deregister(&g->misc);
	g_free(g->dev, g->act_bytes, g->act_cpu, g->act_dma, DMA_TO_DEVICE);
	g_free(g->dev, g->out_bytes, g->out_cpu, g->out_dma, DMA_FROM_DEVICE);
}

#if LINUX_VERSION_CODE >= KERNEL_VERSION(6, 11, 0)
static void gemv_remove(struct platform_device *pdev) { gemv_remove_common(pdev); }
#else
static int gemv_remove(struct platform_device *pdev) { gemv_remove_common(pdev); return 0; }
#endif

static const struct of_device_id gemv_of_match[] = {
	{ .compatible = "xlnx,gemv-q2k-accel-1.0" },
	{ .compatible = "xlnx,gemv-q2k-accel" },
	{ }
};
MODULE_DEVICE_TABLE(of, gemv_of_match);

static struct platform_driver gemv_driver = {
	.probe = gemv_probe,
	.remove = gemv_remove,
	.driver = { .name = DRV_NAME, .of_match_table = gemv_of_match },
};

static struct platform_device *self_pdev;

static int __init gemv_init(void)
{
	struct device_node *np;
	int rc = platform_driver_register(&gemv_driver);

	if (rc)
		return rc;
	np = of_find_matching_node(NULL, gemv_of_match);
	if (np) {
		of_node_put(np);
		return 0;
	}
	{
		struct resource res = DEFINE_RES_MEM(base, REG_SPAN);

		self_pdev = platform_device_register_simple(DRV_NAME, -1, &res, 1);
		if (IS_ERR(self_pdev)) {
			rc = PTR_ERR(self_pdev);
			self_pdev = NULL;
			platform_driver_unregister(&gemv_driver);
			return rc;
		}
	}
	return 0;
}

static void __exit gemv_exit(void)
{
	if (self_pdev)
		platform_device_unregister(self_pdev);
	platform_driver_unregister(&gemv_driver);
}

module_init(gemv_init);
module_exit(gemv_exit);
MODULE_DESCRIPTION("K-quant decode GEMV FPGA accelerator for llama.cpp");
MODULE_LICENSE("GPL");
