use rand::prelude::*;
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};

// -----------------------------------------------------------------------------
// Core Logic: Hamiltonian Mechanics
// -----------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct HmcResult {
    pub samples: Vec<Point>,
    pub acceptance_rate: f64,
}

/// ターゲット分布の種類
pub enum DistType {
    Bimodal, // 二峰性分布
    Banana,  // バナナ型（Rosenbrock）分布
}

impl DistType {
    fn from_str(s: &str) -> Self {
        match s {
            "banana" => DistType::Banana,
            _ => DistType::Bimodal,
        }
    }
}

/// ポテンシャルエネルギー U(q)
fn potential(p: &Point, dist_type: &DistType) -> f64 {
    match dist_type {
        DistType::Bimodal => {
            let d1 = (p.x - 2.5).powi(2) + (p.y - 2.5).powi(2);
            let d2 = (p.x + 2.5).powi(2) + (p.y + 2.5).powi(2);
            // 修正: + 0.0001 を削除し、遠方でポテンシャルが無限大になるようにする（閉じ込めポテンシャル）
            -((-d1 / 1.5).exp() + (-d2 / 1.5).exp()).ln()
        }
        DistType::Banana => (1.0 - p.x).powi(2) + 10.0 * (p.y - p.x.powi(2)).powi(2),
    }
}

/// ポテンシャルエネルギーの勾配 ∇U(q)
fn gradient(p: &Point, dist_type: &DistType) -> Point {
    // 数値微分（中心差分近似）
    let eps = 1e-4;
    let u_x_p = potential(&Point { x: p.x + eps, y: p.y }, dist_type);
    let u_x_m = potential(&Point { x: p.x - eps, y: p.y }, dist_type);
    let u_y_p = potential(&Point { x: p.x, y: p.y + eps }, dist_type);
    let u_y_m = potential(&Point { x: p.x, y: p.y - eps }, dist_type);
    
    Point {
        x: (u_x_p - u_x_m) / (2.0 * eps),
        y: (u_y_p - u_y_m) / (2.0 * eps),
    }
}

/// 運動エネルギー K(p) = p^2 / 2m (m=1とする)
fn kinetic(momentum: &Point) -> f64 {
    0.5 * (momentum.x.powi(2) + momentum.y.powi(2))
}

/// HMCサンプリングのメインロジック
fn run_hmc_chain(
    n_samples: usize,
    step_size: f64,
    num_steps: usize,
    initial_pos: Point,
    dist_name: &str,
) -> HmcResult {
    let mut rng = rand::thread_rng();
    let dist_type = DistType::from_str(dist_name);
    
    let mut current_q = initial_pos;
    let mut samples = Vec::with_capacity(n_samples);
    let mut accepted_count = 0;

    for _ in 0..n_samples {
        // 1. 運動量のサンプリング p ~ N(0, M)
        let current_p = Point {
            x: StandardNormal.sample(&mut rng),
            y: StandardNormal.sample(&mut rng),
        };

        // ハミルトニアンの計算 H = U + K
        let current_u = potential(&current_q, &dist_type);
        let current_k = kinetic(&current_p);
        let current_h = current_u + current_k;

        // 2. リープフロッグ積分
        // --- Velocity Verlet (Standard Leapfrog) ---
        let mut q_lf = current_q.clone();
        let mut p_lf = current_p.clone();
        let mut grad_lf = gradient(&q_lf, &dist_type);

        for _ in 0..num_steps {
            // p half step
            p_lf.x -= 0.5 * step_size * grad_lf.x;
            p_lf.y -= 0.5 * step_size * grad_lf.y;
            
            // q full step
            q_lf.x += step_size * p_lf.x;
            q_lf.y += step_size * p_lf.y;
            
            // p half step
            grad_lf = gradient(&q_lf, &dist_type); // Re-evaluate gradient at new q
            p_lf.x -= 0.5 * step_size * grad_lf.x;
            p_lf.y -= 0.5 * step_size * grad_lf.y;
        }
        // ---------------------------

        // 3. Metropolis Accept/Reject
        let new_u = potential(&q_lf, &dist_type);
        let new_k = kinetic(&p_lf);
        let new_h = new_u + new_k;

        // 判定
        // H_new が無限大（NaN含む）になった場合は、確率0として扱う
        let diff = current_h - new_h;
        let probability = if diff.is_nan() { 0.0 } else { diff.exp() };

        if rng.gen::<f64>() < probability.min(1.0) {
            current_q = q_lf;
            accepted_count += 1;
        }
        
        samples.push(current_q.clone());
    }

    HmcResult {
        samples,
        acceptance_rate: accepted_count as f64 / n_samples as f64,
    }
}

// -----------------------------------------------------------------------------
// Module: Python Interface (PyO3)
// -----------------------------------------------------------------------------
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pyfunction]
fn sample(
    n_samples: usize,
    step_size: f64,
    num_steps: usize,
    start_x: f64,
    start_y: f64,
    dist_type: String
) -> PyResult<(Vec<(f64, f64)>, f64)> {
    let result = run_hmc_chain(
        n_samples, 
        step_size, 
        num_steps, 
        Point { x: start_x, y: start_y }, 
        &dist_type
    );
    
    let py_samples: Vec<(f64, f64)> = result.samples.iter().map(|p| (p.x, p.y)).collect();
    Ok((py_samples, result.acceptance_rate))
}

#[cfg(feature = "python")]
#[pymodule]
fn hamiltonian_sampler_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sample, m)?)?;
    Ok(())
}

// -----------------------------------------------------------------------------
// Module: WebAssembly Interface (wasm-bindgen)
// -----------------------------------------------------------------------------
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn sample_wasm(
    n_samples: usize,
    step_size: f64,
    num_steps: usize,
    start_x: f64,
    start_y: f64,
    dist_type: String
) -> JsValue {
    let result = run_hmc_chain(
        n_samples, 
        step_size, 
        num_steps, 
        Point { x: start_x, y: start_y }, 
        &dist_type
    );
    
    serde_wasm_bindgen::to_value(&result).unwrap()
}