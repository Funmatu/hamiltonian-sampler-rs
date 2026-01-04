import init, { sample_wasm } from './pkg/hamiltonian_sampler_rs.js';

let totalPoints = 0;

async function run() {
    await init(); // WASMのロード

    const canvas = document.getElementById('simCanvas');
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const scale = 40; // 座標系のスケール
    const offsetX = width / 2;
    const offsetY = height / 2;

    // UI Elements
    const distType = document.getElementById('distType');
    const nSamplesRange = document.getElementById('nSamples');
    const stepSizeRange = document.getElementById('stepSize');
    const numStepsRange = document.getElementById('numSteps');
    const runBtn = document.getElementById('runBtn');
    const clearBtn = document.getElementById('clearBtn');
    
    // Display updates
    nSamplesRange.oninput = (e) => document.getElementById('valSamples').innerText = e.target.value;
    stepSizeRange.oninput = (e) => document.getElementById('valStepSize').innerText = e.target.value;
    numStepsRange.oninput = (e) => document.getElementById('valSteps').innerText = e.target.value;

    let currentPos = { x: 0, y: 0 };

    const toCanvas = (x, y) => ({
        cx: offsetX + x * scale,
        cy: offsetY - y * scale // Y軸反転
    });

    const drawGrid = () => {
        ctx.fillStyle = 'rgba(255, 255, 255, 0.05)';
        ctx.fillRect(0, 0, width, height);
        ctx.strokeStyle = '#334155';
        ctx.lineWidth = 0.5;
        
        ctx.beginPath();
        ctx.moveTo(offsetX, 0); ctx.lineTo(offsetX, height);
        ctx.moveTo(0, offsetY); ctx.lineTo(width, offsetY);
        ctx.stroke();
    };

    drawGrid();

    runBtn.onclick = () => {
        const n = parseInt(nSamplesRange.value);
        const eps = parseFloat(stepSizeRange.value);
        const l = parseInt(numStepsRange.value);
        const type = distType.value;

        // Start Timer
        const t0 = performance.now();

        // --- Call Rust WASM ---
        const result = sample_wasm(n, eps, l, currentPos.x, currentPos.y, type);
        // ----------------------

        const t1 = performance.now();
        document.getElementById('compTime').innerText = `${(t1 - t0).toFixed(2)} ms`;

        // Update Stats
        totalPoints += result.samples.length;
        document.getElementById('totalSamples').innerText = totalPoints;
        document.getElementById('accRate').innerText = `${(result.acceptance_rate * 100).toFixed(1)}%`;

        // Update Position for next chain
        const last = result.samples[result.samples.length - 1];
        currentPos = last;

        // Draw Samples
        ctx.fillStyle = 'rgba(129, 140, 248, 0.5)';
        for (let p of result.samples) {
            const { cx, cy } = toCanvas(p.x, p.y);
            ctx.beginPath();
            ctx.arc(cx, cy, 2, 0, Math.PI * 2);
            ctx.fill();
        }
    };

    clearBtn.onclick = () => {
        ctx.clearRect(0, 0, width, height);
        drawGrid();
        totalPoints = 0;
        currentPos = { x: 0, y: 0 };
        document.getElementById('totalSamples').innerText = 0;
    };
}

run();