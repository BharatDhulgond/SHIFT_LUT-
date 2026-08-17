# A Hardware-Efficient Online Softmax Engine for FlashAttention using Base-2 Shift-LUT Exponentiation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![RTL: SystemVerilog](https://img.shields.io/badge/RTL-SystemVerilog-blue.svg)]()
[![FPGA: Spartan--7](https://img.shields.io/badge/FPGA-Spartan--7-orange.svg)]()
[![ASIC: UMC 65nm](https://img.shields.io/badge/ASIC-UMC%2065nm-green.svg)]()

RTL, verification, FPGA, and ASIC implementation of a fully pipelined online-softmax engine that replaces iterative exponentiation with an exact Base-2 decomposition: a 32-entry ROM lookup and a barrel shift.

> **Paper:** *A Hardware-Efficient Online Softmax Engine for FlashAttention using Base-2 Shift-LUT Exponentiation*

---

## Table of Contents

- [Motivation](#motivation)
- [Key Idea](#key-idea)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Implementation Flows](#implementation-flows)
- [Verification](#verification)
- [Results](#results)
- [Contributions](#contributions)
- [Citation](#citation)
- [License](#license)

---

## Motivation

FlashAttention reformulates softmax as an online recurrence that updates a running maximum, a running denominator, and a running weighted output for every incoming attention score. The exponential evaluation therefore sits directly on the critical datapath.

Existing hardware implementations typically rely on hyperbolic CORDIC, polynomial approximation, or piecewise-linear approximation. All three trade area, power, or latency for accuracy, and CORDIC in particular introduces iterative refinement in a stage that is executed once per score.

This work asks whether the exponential can be computed **without iteration**.

## Key Idea

In the online formulation the exponent is always bounded and non-positive, which permits an exact change of base:

$$e^{-\delta} = 2^{-\delta \log_2 e}$$

Writing the scaled exponent as an integer and fractional part, $z = k + f$, gives

$$2^{-z} = 2^{-k} \cdot 2^{-f}$$

| Component | Hardware |
|---|---|
| Integer part $2^{-k}$ | Exact barrel shift |
| Fractional part $2^{-f}$ | 32-entry ROM lookup |

No iterative refinement is required, and the decomposition is exact up to the ROM quantization of the fractional term.

---

## Architecture

The engine is a four-stage pipeline with a BF16 input interface and FP32 internal accumulation.

```
        Input (BF16)
             │
        ┌────▼────┐
        │   S0    │  Input register
        └────┬────┘
        ┌────▼────┐
        │   S1    │  Comparator + BF16 delta generator
        └────┬────┘
        ┌────▼────┐
        │   S2    │  Shift-LUT exponentiation
        └────┬────┘
        ┌────▼────┐
        │   S3    │  Accumulator update
        └────┬────┘
        ┌────▼────┐
        │ FP32    │  Divider
        │ Divide  │
        └────┬────┘
             ▼
      Softmax output
```

**Stage S2 — Shift-LUT exponentiation**

1. Base-2 scaling
2. Saturation
3. BF16 → Q4.5 conversion
4. Integer / fraction split
5. ROM lookup on the fractional part
6. Barrel-shift recombination

---

## Repository Structure

```
.
├── rtl/
│   ├── bf16_compare/            # Running-maximum comparison
│   ├── bf16_delta/              # Delta generation
│   ├── shift_lut_exp/           # Base-2 Shift-LUT exponential unit
│   ├── fp32_mac/                # FP32 multiply-accumulate
│   ├── accumulator_update/      # Online recurrence update
│   ├── fp_div_synth/            # Synthesizable FP32 divider
│   └── softmax_engine_top/      # Top-level integration
│
├── python/
│   ├── softmax_kernel_golden_reference.py
│   └── evaluation.py
│
├── fpga/                        # Vivado project and constraints
├── asic/                        # Genus / Innovus scripts
├── LICENSE
└── README.md
```

---

## Getting Started

### Prerequisites

| Purpose | Tool |
|---|---|
| Simulation | Any SystemVerilog-2012 simulator (Verilator, Questa, Xcelium) |
| FPGA | Xilinx Vivado |
| ASIC | Cadence Genus, Innovus, Voltus |
| Evaluation | Python 3.8+, NumPy, PyTorch, Transformers, Datasets |

### Running the Python evaluation

`evaluation.py` imports `softmax_kernel_golden_reference.py`, so both files must sit in the same directory.

```bash
cd python/
python evaluation.py       # or: python3 evaluation.py
```

This runs functional verification, error analysis, and the full accuracy sweep, and reproduces the metrics reported in the paper.

---

## Implementation Flows

### FPGA — Xilinx Spartan-7 (Vivado)

Synthesis, implementation, timing, utilization, and power reports are included.

### ASIC — UMC 65 nm (Cadence)

The flow covers synthesis (Genus), floorplanning, CTS, and routing (Innovus), with power analysis in Voltus. Signoff timing and power reports are included.

> The UMC 65 nm standard-cell library is not distributed with this repository. A valid foundry/PDK licence is required to rerun the ASIC flow.

---

## Verification

The design is verified at three levels of abstraction.

**RTL** — Module-level testbenches with randomized stimulus and ULP-tolerance checking.

**Functional** — Bit-exact Python golden model compared against RTL output, with waveform validation on mismatches.

**Algorithm** — GPT-2 on WikiText-2, measuring perplexity, top-1 token agreement, and KL divergence against the FP32 baseline.

---

## Results

All comparisons are against a functionally equivalent CORDIC-based implementation of the same pipeline.

### ASIC — UMC 65 nm

| Metric | Shift-LUT | vs. CORDIC |
|---|---|---|
| Total area | 29,561 µm² | 27.9% smaller |
| Exponential unit area | 671 µm² | 18.7× smaller |
| Total power | 4.24 mW | 24.8% lower |
| F<sub>max</sub> | 154.5 MHz | Comparable |

### FPGA — Spartan-7

| Metric | Shift-LUT | vs. CORDIC |
|---|---|---|
| LUTs | 2,050 | 22.8% fewer |
| Exponential unit LUTs | 56 | 13.2× fewer |
| Dynamic power | 69 mW | 10.3% lower |
| F<sub>max</sub> | 107.64 MHz | 6.4% higher |

### Numerical accuracy

Relative to an FP32 baseline:

- Mean relative error ≈ 0.8%
- Maximum relative error ≈ 1.5%
- GPT-2 perplexity within 0.3%
- Top-1 token agreement unchanged
- Low average KL divergence

---

## Contributions

1. The exponent range in FlashAttention's online softmax is fundamentally bounded, which removes the need for a general-purpose exponential unit.
2. Exact Base-2 decomposition maps the exponential onto a small ROM and a barrel shift.
3. The resulting unit is an order of magnitude smaller than a CORDIC exponential at equivalent pipeline throughput.
4. Model-level accuracy is preserved end to end on GPT-2 / WikiText-2.
5. The full flow is released from RTL through FPGA and ASIC signoff.

---

## Citation

```bibtex
@inproceedings{shiftlut_softmax,
  title     = {A Hardware-Efficient Online Softmax Engine for FlashAttention
               using Base-2 Shift-LUT Exponentiation},
  author    = {Bharat Kumar, Harshit Krishna, Prof. Madhav Rao},
  booktitle = {},
  year      = {}
}
```

---

## License

Released under the MIT License. See [LICENSE](LICENSE) for the full text.
