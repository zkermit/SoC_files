# Out-of-tree build of gemv_q2k.ko (source: fpga_drv.c).
#   native:  make
#   cross:   make ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- KDIR=/path/to/kernel/build
# Load:      sudo insmod gemv_q2k.ko                  (binds to the DT node if present)
#            sudo insmod gemv_q2k.ko base=0xA0000000  (no DT node: AXI-Lite base address)
# Weights are pinned zero-copy (no CMA); the default 128 MB CMA is plenty.
obj-m         += gemv_q2k.o
gemv_q2k-objs := fpga_drv.o
ccflags-y     += -I$(src)

KDIR ?= /lib/modules/$(shell uname -r)/build

all:
	$(MAKE) -C $(KDIR) M=$(CURDIR) modules

install:
	$(MAKE) -C $(KDIR) M=$(CURDIR) modules_install

clean:
	$(MAKE) -C $(KDIR) M=$(CURDIR) clean

.PHONY: all install clean
