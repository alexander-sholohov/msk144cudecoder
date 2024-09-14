### MSK144 CUDA Stream Decoder.

**What is it:**

This is CUDA-based MSK144 signal decoder. It accepts input samples in Audio or IQ forms at *stdin*, performs decoding and in case of a message present it prints the result to *stdout*. The algorithm uses GPU power to find the sync pattern, transform the samples into 128 softbits and perform the LDPC decoding. At the final stage, one original CPU-based WSJT Fortran functions is used to decode 77 bits into text message.  
Up to six frames averages are used to find a message in a deep noise.


Audio input format: Mono, 16 bits, signed, 12000 samples per second.  
IQ input format: 8 bits per I/Q, signed(0x80=-128, 0x7f=+127), 12000 samples per second.

Tested on: GTX 730, GTX 1070, NVIDIA Jetson Nano, GTX 1650 Ti & Quadro K620. It should work on any modern NVIDIA GPUs. The Quadra K620 is available for around $20 on eBay.

**How to compile:**

Prereqirements:

```shell
sudo apt-get install build-essential
sudo apt-get install cmake
sudo apt-get install gfortran

```
CUDA toolkit from https://developer.nvidia.com/cuda-downloads

WSJTX repository linked as git submodule. After cloning this repository, execute the following commands:
```shell
cd msk144cudecoder
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

In some cases you may want to invoke `cmake` as
```shell
cmake '-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc' -DCMAKE_CUDA_FLAGS='--gpu-architecture=native' ..
```
Here we specify `CMAKE_CUDA_COMPILER` as the path to `nvcc` on your system. Specifying `CMAKE_CUDA_FLAGS` is optional. By specifying it as shown in the example above, it ensures that the compiled CUDA code can run on the GPU in the machine you compile this on. 


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

Deep scan. More resources usage. Scan in width of 500Hz, use 1Hz step, up to 6 frames average, try to decode frames with unlikely sync pattern.
```shell
./msk144cudecoder --search-step=1 --search-width=500 --scan-depth=6 --nbadsync-threshold=3 
```

Decode provided sample wav file.
```shell
cat ../demo/0001.wav | ./msk144cudecoder
```

Getting IQ stream from rtl_sdr:
```shell
rtl_sdr -s 1920000 -f 144361500 -g 20 - | csdr convert_u8_f  | csdr fir_decimate_cc 8  | csdr fir_decimate_cc 5 | csdr fir_decimate_cc 4 | csdr gain_ff 100.0 | csdr convert_f_s8 | ./msk144cudecoder --search-width=100 --read-mode=2 --scan-depth=3
```

Processing an audio file with `ffmpeg` into the required sample rate and decoding it

```shell
ffmpeg -i myadio.wav -f s16le -acodec pcm_s16le -ar 12000 - 2>/dev/null | msk144cudecoder
```


Links:  
- [WSJT-X Software by Joe K1JT](https://wsjt.sourceforge.io)
- [WSJT Git Repository at sourceforge](https://sourceforge.net/p/wsjt/wsjtx/ci/master/tree/)
- [WSJT CPU based MSK144/JT65/Q65 console decoder](https://github.com/alexander-sholohov/msk144decoder/)
- [CSDR Project](https://github.com/ha7ilm/csdr/)

---

*Acknowledgements to K1JT Joe Taylor and WSJT Development Group. The algorithms, source code, and protocol specifications for the mode MSK144, JT65, Q65 are Copyright © 2001-2021 by one or more of the following authors: Joseph Taylor, K1JT; Bill Somerville, G4WJS; Steven Franke, K9AN; Nico Palermo, IV3NWV; Greg Beam, KI7MT; Michael Black, W9MDB; Edson Pereira, PY2SDR; Philip Karn, KA9Q; and other members of the WSJT Development Group.*

---

Alexander, RA9YER.  
ra9yer@yahoo.com
