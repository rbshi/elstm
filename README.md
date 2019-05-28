# E-LSTM: Efficient Inference of Sparse LSTM on Embedded Heterogeneous System

## Overview
E-LSTM provides an efficient implementation of LSTM inference on the [RISC-V][risc-v] based embedded system. The project contains three blocks as follows,

- [Benchmark](./benchmark): Three fundamental LSTM benchmarks with [MNIST][mnist], [PTB][ptb] and [WikiText][wikitext] dataset respectively. Note that we trained the *sparse* LSTM models in which a portion of weight values are zero. 
- [Software tools](./software): The software scripts, including generator of eSELL sparse representation for LSTM weight matrix, functional simulator and performance fine-tuning scripts. 
- [Hardware tools]: The forked version of official RISC-V toolchains, packeted with a the cycle-level simulator for E-LSTM heterogeneous system. 


## Benchmark
- The three models are implemented on [Pytorch][pytorch] with the [requirements](./benchmark/requirements.txt). 
- Obtain the dataset and pretrained models: 
```bash
$ cd benchmark
$ ./data_download.sh
```
- Evaluate the accuracy:
```bash
# MNIST
$ python MNIST/eval.py -model MNIST/models/nhid:128-nlayer:2-epoch:10.ckpt -ws 0.3 0.5 0.2 0.4 -ht 0.3 0.8 -b 4

# PTB
$ python LM/eval.py --model_path LM/models/PTB/model:LSTM-em:800-nhid:800-nlayers:2-bptt:35-epoch:40-lr:20-tied:False-l1:False-l1_lambda:1e-05-dropout:0.65.ckpt.retrain -m sparsity -ws 0.2 0.5 0.2 0.6 -ht 0.12 0.22 -b 4

# WikiText
$ python LM/eval.py --model_path LM/models/wikitext/model:LSTM-em:1500-nhid:1500-nlayers:2-bptt:35-epoch:20-lr:8.0-tied:False-l1:False-l1_lambda:1e-05-dropout:0.65.ckpt -m sparsity --data LM/data/wikitext --emsize 1500 --nhid 1500 -ws 0.4 0.5 0.3 0.4 -ht 0.28 0.43 -b 4
```
where the arguments represent:
    - `-m` running mode, select the sparsity controlling variable (option: sparsity/threshold)
    - `-data` dataset location
    - `-emsize` embedded length of LSTM input vector
    - `-nhid` size of hidden state
    - `-ws` weight sparsity (for W, U matrix in LSTM-layer1 & LSTM-layer2 respectively)
    - `-ht` threshould for dynamic puring of hidden state (elements in vector **h**)
    - `-b` granularity of hidden state puring (`-b 4` means puring 4 elements in **h** as a whole)
    - other arguments in training and inference phase please refer to `*.py --help`

## Software Tools
- eSELL sparse format construction and comparison ([esell_format.py](./software/esell_format.py)): eSELL is the an sparse matrix representation developed based on [SELL-C-sigma][sell] format, that is used in the E-LSTM system to save the on-chip buffer area cost. 

- Hardware golden reference ([lstm_sim_esell.py](./software/lstm_sim_esell.py)): Package the weight (half-precision float) to 64-bit words for memory storage and CPU-Accelerator interface communication, with the accelerator behavioral simulation. It also dumps the .h file for RISC-V cycle-level simulation.

- Group size fine tuning ([group_size_tuning.py](./software/group_size_tuning.py)): Modeling the accelerator performance for selecting the best factor (N_grp) in cell fusion optimization. The description of input arguments are list by `*.py --help`

## Hardware Tools

The RISC-V toolchain is forked from the [original repo][rocket-chip], and the E-LSTM accelerator cycle-level simulation model is added in. Please reference the following brief to understand the framework and test the accelerator. 

- **IMPORTANT:** Prepare the submodules of [rocket-chip](./rocket-chip) recursively by running `git submodule update --init --recursive` in the root folder. 
- [riscv-tools](./rocket-chip/riscv-tools): all software toolchains and simulator of RISC-V
    - [riscv-tools/riscv-isa-sim](./rocket-chip/riscv-tools/riscv-isa-sim): Spike, a quasi cycle-level simulator of RISC-V system.
        - [riscv-tools/riscv-isa-sim/elstm_rocc](./rocket-chip/riscv-tools/riscv-isa-sim/elstm_rocc): behavioral model of E-LSTM accelerator that coupled with RISC-V via ROCC interface. 
    - Please refer to [Tools' README](./rocket-chip/riscv-tools/README.md) for other toolsets. 
    - For now, please run 'build.sh' to build up all toolsets. 
    - [riscv-tools/riscv-tests/benchmarks/rocc-acc](./rocket-chip/riscv-tools/riscv-tests/benchmarks/rocc-acc): User space test program for ROCC-based accelerator. 
        - [elstm.c](./rocket-chip/riscv-tools/riscv-tests/benchmarks/rocc-acc/elstm.c): host program to simulator the E-LSTM simulator. 
        - Before running the host program, please download the pre-pared header files that store the weight and input activation for each layer (dumped from the benchmark scripts), as follows, 
        ``` bash
        wget https://storage.googleapis.com/rbsharing/elstm/weight_header.tar.gz
        tar -xzvf weight_header.tar.gz
        ```
        - change the included header file and layer corresponding variables in the `elstm.c` for a particular layer; 
        - run `make` to compile the host program for E-LSTM simulation.
        - run `spike --extension=elstm_rocc elstm.riscv` for simulation, the program iteratively processed the 20 input sequences and gives the cycle cost.
        - For the control instruction and workflow details in `elstm.c`, please refere to [elstm-instruction.md](./rocket-chip/riscv-tools/riscv-isa-sim/elstm_rocc/elstm-instruction.md). 























[risc-v]: https://riscv.org
[mnist]: http://yann.lecun.com/exdb/mnist/
[ptb]: https://catalog.ldc.upenn.edu/LDC99T42
[wikitext]: https://www.mediawiki.org/wiki/Wikitext
[pytorch]: https://www.pytorch.org
[sell]: https://arxiv.org/abs/1307.6209
[rocket-chip]: https://github.com/freechipsproject/rocket-chip/tree/7cd3352c3b802c3c50cb864aee828c6106414bb3

