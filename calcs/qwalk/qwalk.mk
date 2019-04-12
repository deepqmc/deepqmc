# This makefile requires variables QWALK_TOP, NWCHEM_TOP, and optionally
# QWALK_PREFIX

QWALK_ARCH ?= Linux-mpi
NWCHEM_ARCH ?= LINUX64
N_BF ?= 1000

QWALK = $(QWALK_TOP)/qwalk-$(QWALK_ARCH)

.SECONDARY:

%_dmc.qw.o: %_dmc.qw %.qw.sys %_opt.qw.wfout
	mpirun -n $(MPI_NODES) $(QWALK) $< | grep --line-buffered "sending walkers" | tqdm --total $(DMC_TOTAL) >/dev/null

%_vmc.qw.o: %_vmc.qw %.qw.sys %_opt.qw.wfout
	$(QWALK) $<

%_opt.qw.wfout: %_opt.qw %.qw.sys %.qw.slater %.qw.jast2
	$(QWALK) $<

%.qw.sys %.qw.slater %.qw.jast2: %.nw.vecs %.nw.out
	$(QWALK_TOP)/converter/nwchem2qmc-$(QWALK_ARCH) -o $*.qw $*

%.nw.vecs: %.nw.movecs
	$(NWCHEM_TOP)/contrib/mov2asc/mov2asc $(N_BF) $^ $@

%.nw.out %.nw.movecs: %.nw.nw
	$(NWCHEM_TOP)/bin/$(NWCHEM_ARCH)/nwchem $^ >$*.nw.out

clean_nwchem:
	rm -f *.nw.aoints.0 *.nw.b *.nw.b^-1 *.nw.c *.nw.cfock *.nw.db *.nw.lagr *.nw.movecs *.nw.oexch *.nw.out *.nw.p *.nw.vecs *.nw.zmat

clean_qwalk_inputs:
	rm -f *.basis *.jast2 *.jast3 *.orb *.slater *.sys

clean_qwalk_outputs:
	rm -f *.config *.config.backup *.json *.o *.wfout

clean: clean_nwchem clean_qwalk_inputs clean_qwalk_outputs

distclean: clean
	rm -f *.qw.log

%: %.o
