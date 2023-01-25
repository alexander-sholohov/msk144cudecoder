### MSK144 CUDA Stream Decoder.

**What is it:**

This is CUDA-based MSK144 signal decoder. It accepts input samples in Audio or IQ forms at *stdin*, performs decoding and in case of a message present it prints the result to *stdout*. The algorithm uses GPU power to find the sync pattern, transform the samples into 128 softbits and perform the LDPC decoding. At the final stage, one original CPU-based WSJT Fortran functions is used to decode 77 bits into text message.  
Up to six frames averages are used to find a message in a deep noise.


Audio input format: Mono, 16 bits, signed, 12000 samples per second.  
IQ input format: 8 bits per I/Q, signed(0x80=-128, 0x7f=+127), 12000 samples per second.

Tested on: GTX 730, GTX 1070, NVIDIA Jetson Nano. It should work on any modern NVIDIA GPUs. 

**How to compile:**

Prereqirements:

```shell
sudo apt-get install build-essential
sudo apt-get install cmake
sudo apt-get install gfortran
sudo apt-get install libboost-dev

```
CUDA toolkit from https://developer.nvidia.com/cuda-downloads

WSJTX repository linked as git submodule. After cloning this repository, execute the following commands:
```shell
cd msk144decoder
git submodule init
git submodule update --progress
```

Commands to build:
```shell
mkdir _build
cd _build
cmake ..
cmake --build . 
```

Executable file *msk144cudecoder* will appear in the current directory.

To run this program on Windows you will need bpdecode.dll. How to build it see ./win32bpdecode/readme.txt


**Run Examples:**

Get brief help:
```shell
./msk144cudecoder  --help
```

Optimal scan. Minimum resources usage. Scan in width of 100Hz, up to 3 frames average.
```shell
./msk144cudecoder --search-width=100 --scan-depth=3
```

Deep scan. More resources usage. Scan in width of 300Hz, up to 6 frames average, try to decode frames with unlikely sync pattern.
```shell
./msk144cudecoder --search-width=300 --scan-depth=6 --nbadsync-threshold=3 
```

Decode provided sample wav file.
```shell
cat ../demo/0001.wav | ./msk144cudecoder
```

Getting IQ stream from rtl_sdr:
```shell
rtl_sdr -s 1920000 -f 144361500 -g 20 - | csdr convert_u8_f  | csdr fir_decimate_cc 8  | csdr fir_decimate_cc 5 | csdr fir_decimate_cc 4 | csdr gain_ff 100.0 | csdr convert_f_s8 | ./msk144cudecoder --search-width=100 --read-mode=2 --scan-depth=3
```


Links:  
- [WSJT-X Software by Joe K1JT](https://physics.princeton.edu/pulsar/k1jt/wsjtx.html)
- [WSJT Git Repository at sourceforge](https://sourceforge.net/p/wsjt/wsjtx/ci/master/tree/)
- [WSJT CPU based MSK144/JT65/Q65 console decoder](https://github.com/alexander-sholohov/msk144decoder/)
- [CSDR Project](https://github.com/ha7ilm/csdr/)

---

*Acknowledgements to K1JT Joe Taylor and WSJT Development Group. The algorithms, source code, and protocol specifications for the mode MSK144, JT65, Q65 are Copyright © 2001-2021 by one or more of the following authors: Joseph Taylor, K1JT; Bill Somerville, G4WJS; Steven Franke, K9AN; Nico Palermo, IV3NWV; Greg Beam, KI7MT; Michael Black, W9MDB; Edson Pereira, PY2SDR; Philip Karn, KA9Q; and other members of the WSJT Development Group.*

---

Alexander, RA9YER.  
ra9yer@yahoo.com
