#include <linux/module.h>
#include <linux/vermagic.h>
#include <linux/compiler.h>

MODULE_INFO(vermagic, VERMAGIC_STRING);
MODULE_INFO(name, KBUILD_MODNAME);

__visible struct module __this_module
__attribute__((section(".gnu.linkonce.this_module"))) = {
	.name = KBUILD_MODNAME,
	.init = init_module,
#ifdef CONFIG_MODULE_UNLOAD
	.exit = cleanup_module,
#endif
	.arch = MODULE_ARCH_INIT,
};

#ifdef RETPOLINE
MODULE_INFO(retpoline, "Y");
#endif

static const struct modversion_info ____versions[]
__used
__attribute__((section("__versions"))) = {
	{ 0xf8cdd757, "module_layout" },
	{ 0x2a63c6dc, "dma_free_attrs" },
	{ 0xedc03953, "iounmap" },
	{ 0x6bc3fbc0, "__unregister_chrdev" },
	{ 0x6ad5f0ce, "__register_chrdev" },
	{ 0x93a219c, "ioremap_nocache" },
	{ 0x100e5057, "dma_alloc_attrs" },
	{ 0xdb7305a1, "__stack_chk_fail" },
	{ 0x7f02188f, "__msecs_to_jiffies" },
	{ 0xa6093a32, "mutex_unlock" },
	{ 0xb44ad4b3, "_copy_to_user" },
	{ 0x15ba50a6, "jiffies" },
	{ 0x69acdf38, "memcpy" },
	{ 0x88db9f48, "__check_object_size" },
	{ 0x41aed6e7, "mutex_lock" },
	{ 0x362ef408, "_copy_from_user" },
	{ 0xbdfb6dbb, "__fentry__" },
	{ 0x5b8239ca, "__x86_return_thunk" },
	{ 0xad27f361, "__warn_printk" },
};

static const char __module_depends[]
__used
__attribute__((section(".modinfo"))) =
"depends=";


MODULE_INFO(srcversion, "D511A077E09A5D7A227B67D");
MODULE_INFO(rhelversion, "8.10");
