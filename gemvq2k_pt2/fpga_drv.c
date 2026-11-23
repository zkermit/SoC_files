// SPDX-License-Identifier: GPL-2.0
/*
 * gemv_q2k: driver for the 4-engine Q2_K decode-GEMV accelerator.
 *
 * Throughput model: Q2_K weights are uploaded once into CMA-backed DMA
 * buffers (per open file, freed on close) and stay resident. A job only
 * copies the activation in (nb * 272 B) and rows * 4 B out, so the FPGA reads
 * each weight byte exactly once per token straight from DDR.
 *
 * Binding: a device-tree node compatible with "xlnx,gemv-q2k-accel-1.0"
 * (default name from the Vivado DT generator) is used if present; otherwise a
 * platform device is created at module parameter `base`, which must match the
 * Vivado address editor.
 */
#include <linux/module.h>
#include <linux/platform_device.h>
#include <linux/of.h>
#include <linux/miscdevice.h>
#include <linux/fs.h>
#include <linux/io.h>
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
static unsigned int max_upload_mb = 768;   /* <= CMA size (Ultra96-V2: cma=640M/768M) */
module_param(max_upload_mb, uint, 0444);
MODULE_PARM_DESC(max_upload_mb, "largest single weight upload in MiB");

/* AXI-Lite register map: must match hls/gemv_q2k_accel.h */
#define REG_CTRL      0x000
#define REG_W(e)      (0x010 + 0x10 * (e))
#define REG_Y(e)      (0x050 + 0x10 * (e))
#define REG_OUT(e)    (0x090 + 0x10 * (e))
#define REG_NB        0x0d0
#define REG_ROWS(e)   (0x0e0 + 0x10 * (e))
#define REG_STATUS    0x120
#define REG_SPAN      0x1000
#define NSHADOW       (0x130 / 4)
#define AP_START      BIT(0)
#define AP_DONE       BIT(1)     /* clear-on-read */
#define AP_IDLE       BIT(2)
#define ST_DONE       2
#define ST_ERROR      4
#define ENGINES       GEMV_Q2K_ENGINES

#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 10, 0)
#define GEMV_NONCOHERENT 1     /* cached CPU mapping + explicit cache sync */
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

struct gemv_wbuf {
	void *cpu;
	dma_addr_t dma;
	size_t size;
};

struct gemv_file {
	struct gemv_dev *g;
	struct mutex lock;
	struct idr bufs;
};

/* ---- DMA helpers ------------------------------------------------------- */
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

/* ---- register access: skip writes whose value is already programmed ---- */
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

/* ---- ioctls ------------------------------------------------------------ */
static long do_info(struct gemv_file *gf, void __user *u)
{
	struct gemv_q2k_info i = {
		.version = GEMV_Q2K_ABI_VERSION,
		.engines = ENGINES,
		.max_blocks = GEMV_Q2K_MAX_BLOCKS,
		.act_block_bytes = GEMV_Q2K_ACT_BLOCK_BYTES,
		.max_rows = max_rows,
		.max_upload_bytes = (u64)max_upload_mb << 20,
	};

	return copy_to_user(u, &i, sizeof(i)) ? -EFAULT : 0;
}

static long do_upload(struct gemv_file *gf, void __user *u)
{
	struct gemv_dev *g = gf->g;
	struct gemv_q2k_upload up;
	struct gemv_wbuf *wb;
	long rc;
	int id;

	if (copy_from_user(&up, u, sizeof(up)))
		return -EFAULT;
	if (!up.bytes || up.bytes > ((u64)max_upload_mb << 20))
		return -EINVAL;
	wb = kzalloc(sizeof(*wb), GFP_KERNEL);
	if (!wb)
		return -ENOMEM;
	wb->size = PAGE_ALIGN(up.bytes);
	wb->cpu = g_alloc(g->dev, wb->size, &wb->dma, DMA_TO_DEVICE);
	if (!wb->cpu) {
		kfree(wb);
		return -ENOMEM;   /* CMA exhausted: user space keeps this tensor on the CPU */
	}
	if (copy_from_user(wb->cpu, u64_to_user_ptr(up.addr), up.bytes)) {
		rc = -EFAULT;
		goto err_free;
	}
	memset((u8 *)wb->cpu + up.bytes, 0, wb->size - up.bytes);
	g_sync_dev(g->dev, wb->dma, wb->size, DMA_TO_DEVICE);

	mutex_lock(&gf->lock);
	id = idr_alloc(&gf->bufs, wb, 1, INT_MAX, GFP_KERNEL);
	mutex_unlock(&gf->lock);
	if (id < 0) {
		rc = id;
		goto err_free;
	}
	up.handle = id;
	if (copy_to_user(u, &up, sizeof(up))) {
		mutex_lock(&gf->lock);
		idr_remove(&gf->bufs, id);
		mutex_unlock(&gf->lock);
		rc = -EFAULT;
		goto err_free;
	}
	return 0;

err_free:
	g_free(g->dev, wb->size, wb->cpu, wb->dma, DMA_TO_DEVICE);
	kfree(wb);
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
	g_free(gf->g->dev, wb->size, wb->cpu, wb->dma, DMA_TO_DEVICE);
	kfree(wb);
	return 0;
}

static long do_submit(struct gemv_file *gf, void __user *u)
{
	struct gemv_dev *g = gf->g;
	struct gemv_q2k_job j;
	struct gemv_wbuf *wb;
	unsigned long deadline;
	size_t abytes, obytes;
	u32 chunk, start = 0, spins = 0;
	long rc = 0;
	int e;

	if (copy_from_user(&j, u, sizeof(j)))
		return -EFAULT;
	if (!j.nb || j.nb > GEMV_Q2K_MAX_BLOCKS || !j.rows || j.rows > max_rows)
		return -EINVAL;
	abytes = (size_t)j.nb * GEMV_Q2K_ACT_BLOCK_BYTES;
	obytes = round_up((size_t)j.rows * 4, 16);

	mutex_lock(&gf->lock);
	wb = idr_find(&gf->bufs, j.handle);
	if (!wb) {
		rc = -ENOENT;
		goto out_file;
	}
	if ((u64)j.rows * j.nb * GEMV_Q2K_BLOCK_BYTES > wb->size) {
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

	/*
	 * Split rows over the engines. Every engine starts on a multiple of 4
	 * rows, so its weight address (4*nb*84 = 336*nb bytes per step) and its
	 * output address are 16-byte aligned for the 128-bit AXI masters.
	 */
	chunk = round_up(DIV_ROUND_UP(j.rows, ENGINES), 4);
	for (e = 0; e < ENGINES; e++) {
		u32 n = start < j.rows ? min(chunk, j.rows - start) : 0;
		u32 s = n ? start : 0;

		wr64(g, REG_W(e), wb->dma + (u64)s * j.nb * GEMV_Q2K_BLOCK_BYTES);
		wr64(g, REG_Y(e), g->act_dma);
		wr64(g, REG_OUT(e), g->out_dma + (u64)s * 4);
		wr32(g, REG_ROWS(e), n);
		start += n;
	}
	wr32(g, REG_NB, j.nb);

	if (!(readl(g->regs + REG_CTRL) & AP_IDLE)) {
		dev_err(g->dev, "accelerator not idle; disabling until reload\n");
		g->broken = true;
		rc = -EBUSY;
		goto out_hw;
	}
	writel(AP_START, g->regs + REG_CTRL);

	/* Jobs last ~0.05-5 ms; spinning has the lowest latency per GEMV. */
	deadline = jiffies + msecs_to_jiffies(j.timeout_ms ? j.timeout_ms : 2000);
	while (!(readl(g->regs + REG_CTRL) & AP_DONE)) {
		if (time_after(jiffies, deadline)) {
			dev_err(g->dev, "job timeout (nb=%u rows=%u); disabling until reload\n",
				j.nb, j.rows);
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
	case GEMV_Q2K_INFO:   return do_info(gf, u);
	case GEMV_Q2K_UPLOAD: return do_upload(gf, u);
	case GEMV_Q2K_FREE:   return do_free(gf, u);
	case GEMV_Q2K_SUBMIT: return do_submit(gf, u);
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
	struct gemv_wbuf *wb = p;
	struct gemv_dev *g = data;

	g_free(g->dev, wb->size, wb->cpu, wb->dma, DMA_TO_DEVICE);
	kfree(wb);
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

/* ---- platform glue ----------------------------------------------------- */
static int gemv_probe(struct platform_device *pdev)
{
	struct gemv_dev *g;
	int rc;

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

	g->misc.minor = MISC_DYNAMIC_MINOR;
	g->misc.name = DRV_NAME;
	g->misc.fops = &gemv_fops;
	g->misc.parent = g->dev;
	/* Safe for unprivileged users: user space never supplies DMA addresses. */
	g->misc.mode = 0666;
	platform_set_drvdata(pdev, g);
	rc = misc_register(&g->misc);
	if (rc)
		goto err;
	dev_info(g->dev, "Q2_K GEMV accelerator: %d engines, max %u blocks/row, /dev/%s\n",
		 ENGINES, GEMV_Q2K_MAX_BLOCKS, DRV_NAME);
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
MODULE_DESCRIPTION("Q2_K decode GEMV FPGA accelerator for llama.cpp");
MODULE_LICENSE("GPL");
