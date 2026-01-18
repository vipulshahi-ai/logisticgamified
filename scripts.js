/**
 * The Logit Lab - Math & Logic Engine
 * ---------------------------------
 * This script handles data generation, logistic regression math, 
 * canvas rendering, and the interactive game state.
 */

class LogitLab {
    constructor() {
        this.canvas = document.getElementById('mainCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.sigmoidCanvas = document.getElementById('sigmoidCanvas');
        this.sCtx = this.sigmoidCanvas.getContext('2d');

        // Model Parameters
        this.w1 = 1.0;
        this.w2 = 1.0;
        this.b = 0.0;

        // Game State
        this.points = [];
        this.level = 1;
        this.resolution = 20; // Heatmap resolution

        this.init();
    }

    init() {
        this.resize();
        this.generateData();
        this.setupEventListeners();
        this.animate();

        window.addEventListener('resize', () => this.resize());
    }

    resize() {
        // Set main canvas size
        const container = this.canvas.parentElement;
        this.canvas.width = container.clientWidth;
        this.canvas.height = container.clientHeight;

        // Set sigmoid canvas size
        this.sigmoidCanvas.width = this.sigmoidCanvas.parentElement.clientWidth;
        this.sigmoidCanvas.height = 120;
    }

    generateData() {
        this.points = [];
        const numPoints = 60;

        // Create two clusters
        // Cluster 0 (Red/Failure) - Centered at roughly (30%, 70%)
        for (let i = 0; i < numPoints / 2; i++) {
            this.points.push({
                x: 0.2 + Math.random() * 0.3,
                y: 0.6 + Math.random() * 0.3,
                label: 0
            });
        }

        // Cluster 1 (Blue/Success) - Centered at roughly (70%, 30%)
        for (let i = 0; i < numPoints / 2; i++) {
            this.points.push({
                x: 0.5 + Math.random() * 0.3,
                y: 0.1 + Math.random() * 0.3,
                label: 1
            });
        }
    }

    setupEventListeners() {
        const w1Slider = document.getElementById('w1-slider');
        const w2Slider = document.getElementById('w2-slider');
        const bSlider = document.getElementById('b-slider');
        const autoBtn = document.getElementById('auto-train');
        const resetBtn = document.getElementById('reset-data');
        const tabBtns = document.querySelectorAll('.tab-btn');

        const updateParams = () => {
            this.w1 = parseFloat(w1Slider.value);
            this.w2 = parseFloat(w2Slider.value);
            this.b = parseFloat(bSlider.value);
            this.updateUI();
        };

        w1Slider.addEventListener('input', updateParams);
        w2Slider.addEventListener('input', updateParams);
        bSlider.addEventListener('input', updateParams);

        autoBtn.addEventListener('click', () => this.runGradientDescent());
        resetBtn.addEventListener('click', () => {
            this.generateData();
            this.updateUI();
        });

        tabBtns.forEach(btn => {
            btn.addEventListener('click', (e) => {
                tabBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.level = parseInt(btn.dataset.level);
                this.updateModuleInfo();
            });
        });
    }

    /**
     * Logistic Regression Math
     */
    sigmoid(z) {
        return 1 / (1 + Math.exp(-z));
    }

    predict(x, y) {
        // Normalize coordinates to [-5, 5] for better math stability in the viz
        const nx = (x - 0.5) * 10;
        const ny = (y - 0.5) * 10;
        const z = this.w1 * nx + this.w2 * ny + this.b;
        return this.sigmoid(z);
    }

    calculateMetrics() {
        let correct = 0;
        let totalSquaredError = 0;

        this.points.forEach(p => {
            const prob = this.predict(p.x, p.y);

            // Accuracy
            const predLabel = prob >= 0.5 ? 1 : 0;
            if (predLabel === p.label) correct++;

            // Mean Squared Error: (pred - actual)^2
            totalSquaredError += Math.pow(prob - p.label, 2);
        });

        const accuracy = (correct / this.points.length) * 100;
        const mse = totalSquaredError / this.points.length;

        return { accuracy, mse };
    }

    /**
     * Automated Optimization (Gradient Descent)
     */
    async runGradientDescent() {
        const learningRate = 0.5;
        const epochs = 50;

        for (let i = 0; i < epochs; i++) {
            let dw1 = 0;
            let dw2 = 0;
            let db = 0;

            this.points.forEach(p => {
                const nx = (p.x - 0.5) * 10;
                const ny = (p.y - 0.5) * 10;
                const prob = this.predict(p.x, p.y);
                const error = prob - p.label;

                dw1 += error * nx;
                dw2 += error * ny;
                db += error;
            });

            const m = this.points.length;
            this.w1 -= (learningRate * dw1) / m;
            this.w2 -= (learningRate * dw2) / m;
            this.b -= (learningRate * db) / m;

            // Sync sliders
            document.getElementById('w1-slider').value = this.w1;
            document.getElementById('w2-slider').value = this.w2;
            document.getElementById('b-slider').value = this.b;

            this.updateUI();
            await new Promise(r => setTimeout(r, 20)); // Animation delay
        }
    }

    /**
     * Rendering Logic
     */
    drawHeatmap() {
        const w = this.canvas.width;
        const h = this.canvas.height;
        const res = this.resolution;

        for (let x = 0; x < w; x += res) {
            for (let y = 0; y < h; y += res) {
                const prob = this.predict(x / w, y / h);

                // Blue for Success (1), Red for Failure (0)
                // We use opacity or color interpolation
                const blue = Math.floor(prob * 150);
                const red = Math.floor((1 - prob) * 150);

                this.ctx.fillStyle = `rgba(${red}, ${blue * 0.5}, ${blue}, 0.2)`;
                this.ctx.fillRect(x, y, res, res);
            }
        }
    }

    drawBoundary() {
        const w = this.canvas.width;
        const h = this.canvas.height;

        this.ctx.beginPath();
        this.ctx.strokeStyle = '#00ff88';
        this.ctx.lineWidth = 3;
        this.ctx.setLineDash([10, 5]);

        // We want to find the line where w1*nx + w2*ny + b = 0
        // nx = (x/w - 0.5)*10 => x/w = nx/10 + 0.5 => x = w * (nx/10 + 0.5)
        // ny = - (w1*nx + b) / w2

        let started = false;
        for (let nx = -5; nx <= 5; nx += 0.1) {
            const ny = - (this.w1 * nx + this.b) / this.w2;
            if (ny >= -5 && ny <= 5) {
                const canvasX = (nx / 10 + 0.5) * w;
                const canvasY = (ny / 10 + 0.5) * h;

                if (!started) {
                    this.ctx.moveTo(canvasX, canvasY);
                    started = true;
                } else {
                    this.ctx.lineTo(canvasX, canvasY);
                }
            }
        }
        this.ctx.stroke();
        this.ctx.setLineDash([]);

        // Add glow
        this.ctx.shadowBlur = 15;
        this.ctx.shadowColor = '#00ff88';
        this.ctx.stroke();
        this.ctx.shadowBlur = 0;
    }

    drawPoints() {
        const w = this.canvas.width;
        const h = this.canvas.height;

        this.points.forEach(p => {
            this.ctx.beginPath();
            this.ctx.arc(p.x * w, p.y * h, 6, 0, Math.PI * 2);

            if (p.label === 1) {
                this.ctx.fillStyle = '#00d4ff';
                this.ctx.shadowColor = '#00d4ff';
            } else {
                this.ctx.fillStyle = '#ff4757';
                this.ctx.shadowColor = '#ff4757';
            }

            this.ctx.shadowBlur = 10;
            this.ctx.fill();
            this.ctx.strokeStyle = 'white';
            this.ctx.lineWidth = 1;
            this.ctx.stroke();
            this.ctx.shadowBlur = 0;
        });
    }

    drawSigmoidPanel() {
        const sw = this.sigmoidCanvas.width;
        const sh = this.sigmoidCanvas.height;
        this.sCtx.clearRect(0, 0, sw, sh);

        // Draw Axes
        this.sCtx.strokeStyle = 'rgba(255,255,255,0.1)';
        this.sCtx.beginPath();
        this.sCtx.moveTo(0, sh / 2); this.sCtx.lineTo(sw, sh / 2);
        this.sCtx.moveTo(sw / 2, 0); this.sCtx.lineTo(sw / 2, sh);
        this.sCtx.stroke();

        // Draw Sigmoid Curve
        this.sCtx.beginPath();
        this.sCtx.strokeStyle = '#00d4ff';
        this.sCtx.lineWidth = 2;
        for (let x = 0; x < sw; x++) {
            const z = ((x / sw) - 0.5) * 10;
            const y = this.sigmoid(z);
            const canvasY = sh - (y * sh);
            if (x === 0) this.sCtx.moveTo(x, canvasY);
            else this.sCtx.lineTo(x, canvasY);
        }
        this.sCtx.stroke();

        // Draw moving dot (mean point z value)
        // We'll just use a fixed z from w1,w2,b at some sample point or just based on sliders for demo
        const demoZ = (this.w1 + this.w2) / 2 + this.b; // Just a proxy for "current state"
        const dotX = (demoZ / 10 + 0.5) * sw;
        const dotY = sh - (this.sigmoid(demoZ) * sh);

        this.sCtx.beginPath();
        this.sCtx.arc(Math.max(0, Math.min(sw, dotX)), Math.max(0, Math.min(sh, dotY)), 5, 0, Math.PI * 2);
        this.sCtx.fillStyle = '#00ff88';
        this.sCtx.fill();
        this.sCtx.shadowBlur = 10;
        this.sCtx.shadowColor = '#00ff88';
        this.sCtx.stroke();
    }

    updateUI() {
        const { accuracy, mse } = this.calculateMetrics();

        // Update labels
        document.querySelectorAll('.w1-val').forEach(el => el.textContent = this.w1.toFixed(1));
        document.querySelectorAll('.w2-val').forEach(el => el.textContent = this.w2.toFixed(1));
        document.querySelectorAll('.b-val').forEach(el => el.textContent = this.b.toFixed(1));

        // Update Meters
        const accFill = document.getElementById('accuracy-fill');
        const accText = document.getElementById('accuracy-text');
        if (accFill) accFill.style.width = `${accuracy}%`;
        if (accText) accText.textContent = `${accuracy.toFixed(0)}%`;

        const lossFill = document.getElementById('loss-fill');
        const lossText = document.getElementById('loss-text');
        const errorPercent = Math.min(100, mse * 100);
        if (lossFill) lossFill.style.width = `${errorPercent}%`;
        if (lossText) lossText.textContent = mse.toFixed(2);

        // Update Formula
        const formula = document.getElementById('formula-display');
        if (formula) formula.innerHTML = `z = (${this.w1.toFixed(1)})x₁ + (${this.w2.toFixed(1)})x₂ + (${this.b.toFixed(1)})`;
    }

    updateModuleInfo() {
        const title = document.getElementById('module-title');
        const desc = document.getElementById('module-desc');

        switch (this.level) {
            case 1:
                title.textContent = "The Linear Core";
                desc.innerHTML = "Every model starts with math: <strong>z = w₁x₁ + w₂x₂ + b</strong>. This formula gives us a raw score. Higher <em>z</em> means we're leaning towards Success.";
                break;
            case 2:
                title.textContent = "The S-Curve Squash";
                desc.innerHTML = "We take the raw score <em>z</em> and squash it through the <strong>Sigmoid function</strong>. This converts any number into a probability between 0 and 1!";
                break;
            case 3:
                title.textContent = "Why a Linear Boundary?";
                desc.innerHTML = "Even though the Sigmoid is curved, we draw the line exactly where the probability is <strong>0.5</strong>. Since this happens when <em>z = 0</em>, the resulting boundary is a perfectly straight line!";
                break;
        }

        // Show quiz for the current level
        if (this.quizManager) {
            this.quizManager.startLevelQuiz(this.level);
        }
    }

    animate() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        this.drawHeatmap();
        this.drawBoundary();
        this.drawPoints();
        this.drawSigmoidPanel();

        requestAnimationFrame(() => this.animate());
    }
}

class QuizManager {
    constructor(logitLab) {
        this.logitLab = logitLab;
        this.container = document.getElementById('quiz-container');
        this.questionEl = document.getElementById('quiz-question');
        this.optionsEl = document.getElementById('quiz-options');
        this.progressEl = document.getElementById('quiz-progress');
        this.feedbackEl = document.getElementById('quiz-feedback');
        this.nextBtn = document.getElementById('quiz-next');

        this.nextBtn.onclick = () => this.goToNextQuestion();

        this.currentQuiz = [];
        this.currentIndex = 0;

        this.quizzes = {
            1: [
                {
                    q: "What happens to the boundary if you increase the Bias (b)?",
                    options: ["It rotates", "It shifts its position", "It disappears", "It becomes curved"],
                    correct: 1,
                    hint: "Bias is like an offset; it moves the whole line without changing its tilt."
                },
                {
                    q: "If z = w₁.x₁ + w₂.x₂ + b, what is z called?",
                    options: ["Probability", "The Linear Combo (Logit)", "The Error", "The Sigmoid"],
                    correct: 1,
                    hint: "z is the raw score before we apply any 'squashing' function."
                }
            ],
            2: [
                {
                    q: "What is the range of the Sigmoid function σ(z)?",
                    options: ["-1 to 1", "0 to 100", "0 to 1", "Negative infinity to infinity"],
                    correct: 2,
                    hint: "Sigmoid squashes everything into a standard probability range."
                },
                {
                    q: "If z is a very large positive number, what will σ(z) be closest to?",
                    options: ["0", "0.5", "1", "-1"],
                    correct: 2,
                    hint: "A high positive score means high confidence in 'Success' (1)."
                }
            ],
            3: [
                {
                    q: "Where exactly is the 'Decision Boundary' drawn?",
                    options: ["At P = 0", "At P = 0.5", "At P = 1", "Wherever you want"],
                    correct: 1,
                    hint: "The boundary is the tipping point where you switch from Class 0 to Class 1."
                },
                {
                    q: "If the Sigmoid is curved, why is the boundary a straight line?",
                    options: ["Because math is weird", "Because z = 0 is a linear equation", "The canvas is flat", "The weights are constant"],
                    correct: 1,
                    hint: "The curve happens in 'probability space', but the split happens where the linear math hits 0."
                }
            ]
        };
    }

    startLevelQuiz(level) {
        this.currentQuiz = this.quizzes[level];
        this.currentIndex = 0;
        this.container.classList.remove('invisible');
        this.showQuestion();
    }

    showQuestion() {
        const item = this.currentQuiz[this.currentIndex];
        this.questionEl.textContent = item.q;
        this.progressEl.textContent = `${this.currentIndex + 1}/${this.currentQuiz.length}`;
        this.feedbackEl.classList.add('hidden');
        this.nextBtn.classList.add('hidden');

        this.optionsEl.innerHTML = '';
        item.options.forEach((opt, idx) => {
            const btn = document.createElement('button');
            btn.className = 'quiz-opt';
            btn.textContent = opt;
            btn.onclick = () => this.checkAnswer(idx);
            this.optionsEl.appendChild(btn);
        });
    }

    checkAnswer(idx) {
        const item = this.currentQuiz[this.currentIndex];
        const buttons = this.optionsEl.querySelectorAll('.quiz-opt');

        buttons.forEach(b => b.disabled = true);

        if (idx === item.correct) {
            buttons[idx].classList.add('correct');
            this.showFeedback(true, "Correct! " + item.hint);
            this.nextBtn.classList.remove('hidden');

            if (this.currentIndex === this.currentQuiz.length - 1) {
                this.nextBtn.textContent = "Finish Quiz";
            } else {
                this.nextBtn.textContent = "Next Question →";
            }
        } else {
            buttons[idx].classList.add('incorrect');
            this.showFeedback(false, "Try again! Hint: " + item.hint);
            setTimeout(() => {
                buttons.forEach(b => {
                    b.disabled = false;
                    b.classList.remove('incorrect');
                });
            }, 2000);
        }
    }

    goToNextQuestion() {
        this.currentIndex++;
        if (this.currentIndex < this.currentQuiz.length) {
            this.showQuestion();
        } else {
            this.showFeedback(true, "Level complete! You've mastered these concepts.");
            this.nextBtn.classList.add('hidden');
        }
    }

    showFeedback(isCorrect, msg) {
        this.feedbackEl.textContent = msg;
        this.feedbackEl.className = `feedback-msg ${isCorrect ? 'correct' : 'incorrect'}`;
        this.feedbackEl.classList.remove('hidden');
    }
}

// Start the Lab
document.addEventListener('DOMContentLoaded', () => {
    const lab = new LogitLab();
    lab.quizManager = new QuizManager(lab);
    lab.updateModuleInfo(); // Trigger initial quiz
});
