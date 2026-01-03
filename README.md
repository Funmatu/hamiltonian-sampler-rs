# Hamiltonian Sampler RS: Dual-Runtime MCMC Engine

![Build Status](https://github.com/Funmatu/hamiltonian-sampler-rs/actions/workflows/deploy.yml/badge.svg)
![Rust](https://img.shields.io/badge/Language-Rust-orange.svg)
![Platform](https://img.shields.io/badge/Platform-WASM%20%7C%20Python-blue.svg)

**Hamiltonian Sampler RS** is a high-performance, strictly typed implementation of Hamiltonian Monte Carlo (HMC) algorithms. Designed for R&D in Physical AI and Robotics, it adopts a "Dual-Runtime" architecture, allowing the exact same core logic to be deployed as a **Python Native Extension** (for rigorous backend analysis) and a **WebAssembly Module** (for interactive frontend visualization).

## 1. Mathematical Formulation

HMC utilizes Hamiltonian dynamics to propose samples that follow a target probability distribution $P(q) \propto e^{-U(q)}$. It introduces auxiliary momentum variables $p$ and simulates the evolution of the system.

### Hamiltonian
The total energy (Hamiltonian) of the system is defined as:
$$H(q, p) = U(q) + K(p)$$
Where:
* $U(q)$: Potential energy (negative log-probability of the target distribution).
* $K(p) = \frac{1}{2}p^T M^{-1} p$: Kinetic energy (usually Gaussian).

### Leapfrog Integrator
To solve the equations of motion numerically while preserving volume (symplecticity) and reversibility, we employ the Leapfrog integration scheme:

1.  **Half-step Momentum Update:**
    $$p(t + \epsilon/2) = p(t) - \frac{\epsilon}{2} \nabla U(q(t))$$
2.  **Full-step Position Update:**
    $$q(t + \epsilon) = q(t) + \epsilon M^{-1} p(t + \epsilon/2)$$
3.  **Half-step Momentum Update:**
    $$p(t + \epsilon) = p(t + \epsilon/2) - \frac{\epsilon}{2} \nabla U(q(t + \epsilon))$$

This physics-based approach allows the sampler to traverse long distances in the state space, efficiently exploring complex distributions like high-dimensional gaussians or "Banana" shapes.

## 2. Architecture

This repository uses Cargo features to compile the core Rust logic into two distinct targets:

| Feature | Target | Technology | Use Case |
| :--- | :--- | :--- | :--- |
| `python` | `.so` / `.pyd` | **PyO3** | High-performance backend sampling, integration with NumPy/PyTorch. |
| `wasm` | `.wasm` | **wasm-bindgen** | Client-side visualization, interactive demos on GitHub Pages. |

```mermaid
graph TD
    subgraph "Core Logic (Rust)"
        HMC[HmcSystem]
        Pot[Potential U(q)]
        Grad[Gradient ∇U(q)]
        LF[Leapfrog Integrator]
    end

    subgraph "Interface Layer"
        PyBind[PyO3 Bindings]
        WasmBind[wasm-bindgen]
    end

    subgraph "Runtimes"
        PyEnv[Python Script]
        Browser[Web Browser]
    end

    HMC --> Pot
    HMC --> LF
    LF --> Grad

    HMC --> PyBind --> PyEnv
    HMC --> WasmBind --> Browser
```

## 3. Installation & Usage

### A. Python (for Analysis)

Prerequisites: Rust toolchain, Python 3.8+, `maturin`.

```bash
# 1. Install maturin
pip install maturin

# 2. Build and install locally
maturin develop --release --features python

# 3. Run in Python
python -c "
import hamiltonian_sampler_rs as hmc
import numpy as np

# Sample from a Banana distribution
# (n_samples, step_size, num_steps, start_x, start_y, dist_type)
samples, acceptance_rate = hmc.sample(10000, 0.1, 20, 0.0, 0.0, 'banana')

print(f'Acceptance Rate: {acceptance_rate:.2%}')
print(f'Last Sample: {samples[-1]}')
"
```

### B. WebAssembly (for Visualization)

Prerequisites: `wasm-pack`.

```bash
# 1. Build WASM artifact
wasm-pack build --target web --out-dir www/pkg --no-default-features --features wasm

# 2. Serve locally
cd www
python3 -m http.server 8000
```

## 4. Performance Benchmarks

*Hardware: MacBook Pro M2, Single Core*

| Metric | Python (Native Pure) | Rust (via PyO3) | Speedup |
| :--- | :--- | :--- | :--- |
| 1M Samples (Bimodal) | ~12.5 s | ~0.45 s | **~27x** |

*Note: The Rust implementation benefits from aggressive compiler optimizations (SIMD) and zero-overhead memory management.*

## 5. Development

* **Edit Core Logic:** Modify `src/lib.rs`.
* **Test Python:** `maturin develop --features python && python test_script.py`
* **Test WASM:** `wasm-pack build --features wasm` and refresh browser.

## License

MIT