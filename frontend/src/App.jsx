import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  AlertCircle,
  ArrowUpRight,
  BookOpen,
  BrainCircuit,
  CheckCircle2,
  ChevronRight,
  CircleHelp,
  Clock3,
  Cpu,
  Database,
  Dna,
  FileImage,
  FlaskConical,
  Gauge,
  GitBranch,
  Info,
  Layers3,
  Microscope,
  Network,
  RotateCcw,
  Sparkles,
  Target,
  Terminal,
  Trophy,
  Upload,
  Zap
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";

const API = "";

const navItems = [
  { id: "studio", label: "Live Studio", icon: Sparkles },
  { id: "experiment", label: "Experiment Lab", icon: Activity },
  { id: "algorithm", label: "Algorithm Guide", icon: Dna },
  { id: "research", label: "Research Notes", icon: BookOpen }
];

const percent = (value) => `${(Number(value || 0) * 100).toFixed(2)}%`;
const number = (value) => Number(value || 0).toLocaleString("en-IN");

function StatCard({ icon: Icon, label, value, detail, tone = "cyan" }) {
  return (
    <div className={`stat-card tone-${tone}`}>
      <div className="stat-top">
        <div className="stat-icon"><Icon size={17} /></div>
        <span>{label}</span>
      </div>
      <div className="stat-value">{value}</div>
      <div className="stat-detail">{detail}</div>
    </div>
  );
}

function SectionHeading({ eyebrow, title, description, right }) {
  return (
    <div className="section-heading">
      <div>
        <div className="eyebrow">{eyebrow}</div>
        <h2>{title}</h2>
        {description && <p>{description}</p>}
      </div>
      {right}
    </div>
  );
}

function EmptyTraining({ status, onRefresh }) {
  const failed = status.status === "failed";
  return (
    <div className="empty-training">
      <div className="empty-orbit"><BrainCircuit size={36} /></div>
      <div className="eyebrow">MODEL NOT READY</div>
      <h2>{failed ? "Training needs attention" : "Your experiment has not been trained yet"}</h2>
      <p>{failed ? status.error || "Check the terminal output and correct the missing dependency or dataset issue." : "Run the training command in the VS Code terminal. The model, charts and MNIST examples will appear here when training finishes."}</p>
      <div className="terminal-snippet">
        <div className="terminal-bar"><span /><span /><span /><small>VS CODE TERMINAL</small></div>
        <code>python train.py</code>
      </div>
      <button className="button button-primary" onClick={onRefresh}><RotateCcw size={15} /> Refresh training status</button>
      <div className="empty-note"><Info size={14} /> First run downloads MNIST once. Keep an internet connection during the download.</div>
    </div>
  );
}

function TrainingBanner({ status }) {
  if (!status || status.status === "not_started" || status.status === "completed") return null;
  const failed = status.status === "failed";
  return (
    <div className={`training-banner ${failed ? "is-failed" : "is-running"}`}>
      <div className="banner-icon">{failed ? <AlertCircle size={18} /> : <Activity size={18} />}</div>
      <div className="banner-copy">
        <strong>{failed ? "Training stopped" : status.stage || "Training in progress"}</strong>
        <span>{failed ? status.error : "Watch the experiment progress in the terminal, then refresh this page when it finishes."}</span>
        {!failed && <div className="progress-track"><div style={{ width: `${status.percent || 0}%` }} /></div>}
      </div>
      <b>{Math.round(status.percent || 0)}%</b>
    </div>
  );
}

function LiveStudio({ dashboard, examples, prediction, setPrediction, onPredict, busy, error, setError }) {
  const canvasRef = useRef(null);
  const drawingRef = useRef(false);
  const [selectedSample, setSelectedSample] = useState(null);
  const [uploadedName, setUploadedName] = useState("");

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    context.fillStyle = "#050913";
    context.fillRect(0, 0, canvas.width, canvas.height);
  }, []);

  const getPoint = (event) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    return {
      x: (event.clientX - rect.left) * canvas.width / rect.width,
      y: (event.clientY - rect.top) * canvas.height / rect.height
    };
  };

  const startDrawing = (event) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.setPointerCapture(event.pointerId);
    const context = canvas.getContext("2d");
    const point = getPoint(event);
    context.beginPath();
    context.moveTo(point.x, point.y);
    context.strokeStyle = "#f8fbff";
    context.lineWidth = 30;
    context.lineCap = "round";
    context.lineJoin = "round";
    context.lineTo(point.x + 0.5, point.y + 0.5);
    context.stroke();
    drawingRef.current = true;
  };

  const continueDrawing = (event) => {
    if (!drawingRef.current) return;
    const context = canvasRef.current.getContext("2d");
    const point = getPoint(event);
    context.lineTo(point.x, point.y);
    context.stroke();
  };

  const stopDrawing = () => {
    drawingRef.current = false;
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const context = canvas.getContext("2d");
    context.fillStyle = "#050913";
    context.fillRect(0, 0, canvas.width, canvas.height);
    setPrediction(null);
    setSelectedSample(null);
    setUploadedName("");
    setError("");
  };

  const analyseCanvas = () => {
    canvasRef.current.toBlob(async (blob) => {
      if (!blob) return;
      const file = new File([blob], "drawn-digit.png", { type: "image/png" });
      await onPredict(file);
      setUploadedName("Canvas drawing");
      setSelectedSample(null);
    }, "image/png");
  };

  const chooseSample = async (sample) => {
    setSelectedSample(sample.id);
    setUploadedName(`MNIST test example · actual digit ${sample.label}`);
    const response = await fetch(`data:image/png;base64,${sample.image}`);
    const blob = await response.blob();
    const file = new File([blob], `mnist-digit-${sample.label}.png`, { type: "image/png" });
    await onPredict(file);
  };

  const chooseUpload = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setUploadedName(file.name);
    setSelectedSample(null);
    await onPredict(file);
    event.target.value = "";
  };

  const metadata = dashboard?.metadata;
  const bestGenome = metadata?.best_genome;
  const samplePicks = examples;

  return (
    <>
      <SectionHeading
        eyebrow="INTERACTIVE INFERENCE"
        title="Give the network a digit."
        description="Draw one handwritten digit, upload an image, or test a real example from the MNIST test split."
        right={<div className="trained-pill"><span /> MODEL TRAINED</div>}
      />
      <div className="studio-grid">
        <div className="panel drawing-panel">
          <div className="panel-topline">
            <div><span className="panel-kicker">01 / INPUT</span><h3>Digit canvas</h3></div>
            <span className="mini-tag">28 × 28 model input</span>
          </div>
          <div className="canvas-wrap">
            <div className="canvas-grid" />
            <canvas
              ref={canvasRef}
              width={360}
              height={360}
              onPointerDown={startDrawing}
              onPointerMove={continueDrawing}
              onPointerUp={stopDrawing}
              onPointerCancel={stopDrawing}
              onPointerLeave={stopDrawing}
              aria-label="Draw a handwritten digit"
            />
            <div className="canvas-hint">DRAW HERE</div>
          </div>
          <div className="canvas-actions">
            <button className="button button-quiet" onClick={clearCanvas}><RotateCcw size={15} /> Clear canvas</button>
            <button className="button button-primary" onClick={analyseCanvas} disabled={busy}><Sparkles size={15} /> {busy ? "Analysing…" : "Analyse digit"}</button>
          </div>
          <div className="upload-row">
            <label className="upload-control"><Upload size={15} /> Upload PNG / JPG<input type="file" accept="image/png,image/jpeg,image/webp" onChange={chooseUpload} /></label>
            <span className="input-filename">{uploadedName || "No input selected"}</span>
          </div>
          <div className="sample-divider"><span>OR TRY A REAL MNIST TEST IMAGE</span></div>
          <div className="sample-row">
            {samplePicks.map((sample) => (
              <button key={sample.id} className={`sample-chip ${selectedSample === sample.id ? "selected" : ""}`} onClick={() => chooseSample(sample)} disabled={busy}>
                <img src={`data:image/png;base64,${sample.image}`} alt={`MNIST digit ${sample.label}`} />
                <span>#{sample.label}</span>
              </button>
            ))}
          </div>
          <div className="input-tip"><Info size={14} /><span>Best input: one bold digit, centered, with dark background and light strokes. Avoid words, multiple digits or scenery.</span></div>
          {error && <div className="error-line"><AlertCircle size={15} /> {error}</div>}
        </div>

        <div className="panel prediction-panel">
          <div className="panel-topline">
            <div><span className="panel-kicker">02 / MODEL OUTPUT</span><h3>Prediction analysis</h3></div>
            <span className="live-indicator"><span /> LIVE</span>
          </div>
          {!prediction ? (
            <div className="prediction-empty">
              <div className="prediction-empty-icon"><Target size={27} /></div>
              <h4>Waiting for input</h4>
              <p>Your prediction, class probabilities and model analysis will appear here after you submit a digit.</p>
              <div className="empty-steps"><span>01 · Draw or upload</span><span>02 · CNN inference</span><span>03 · Read the analysis</span></div>
            </div>
          ) : (
            <div className="prediction-result">
              <div className="result-hero">
                <div className="processed-preview"><img src={`data:image/png;base64,${prediction.normalized_image}`} alt="Normalized 28 by 28 model input" /></div>
                <div className="result-main">
                  <span className="result-label">PREDICTED DIGIT</span>
                  <div className="predicted-digit">{prediction.prediction}</div>
                  <div className="confidence-line"><strong>{prediction.confidence_percentage}%</strong><span>{prediction.confidence_label}</span></div>
                </div>
              </div>
              <div className="result-metrics">
                <div><span>Inference time</span><strong>{prediction.inference_ms} ms</strong></div>
                <div><span>Top-two gap</span><strong>{prediction.top_two_gap} pp</strong></div>
                <div><span>Input quality</span><strong>{prediction.input_quality}</strong></div>
              </div>
              <div className="probability-block">
                <div className="subheading-row"><h4>Top class scores</h4><span>SOFTMAX OUTPUT</span></div>
                {prediction.top_three.map((item, index) => (
                  <div className="probability-row" key={item.digit}>
                    <b>{item.digit}</b>
                    <div className="probability-track"><div className={index === 0 ? "top" : ""} style={{ width: `${item.percentage}%` }} /></div>
                    <span>{item.percentage.toFixed(2)}%</span>
                  </div>
                ))}
              </div>
              <div className="analysis-block">
                <div className="subheading-row"><h4>What the model observed</h4><span><Microscope size={12} /> INPUT ANALYSIS</span></div>
                {prediction.analysis.map((line, index) => <p key={index}><span>{String(index + 1).padStart(2, "0")}</span>{line}</p>)}
              </div>
              <div className="compute-strip">
                <div><Cpu size={16} /><span>Compute</span><b>{prediction.compute.compute_device}</b></div>
                <div><Network size={16} /><span>Parameters</span><b>{number(prediction.parameter_count)}</b></div>
                <div><Zap size={16} /><span>Est. MACs / image</span><b>{number(prediction.estimated_macs)}</b></div>
              </div>
            </div>
          )}
        </div>
      </div>

      {prediction && (
        <div className="panel pipeline-panel">
          <div className="panel-topline"><div><span className="panel-kicker">03 / INFERENCE TRACE</span><h3>How this input travelled through the model</h3></div><span className="mini-tag">Selected GA-LeNet architecture</span></div>
          <div className="pipeline-grid">
            {prediction.pipeline.map((stage, index) => (
              <div className="pipeline-step" key={stage.name}>
                <div className="pipeline-number">{String(index + 1).padStart(2, "0")}</div>
                <strong>{stage.name}</strong>
                <span>{stage.shape}</span>
                <p>{stage.purpose}</p>
                {index < prediction.pipeline.length - 1 && <ChevronRight className="pipeline-arrow" size={16} />}
              </div>
            ))}
          </div>
          <div className="explainability-note"><Info size={15} /><span>The probability panel explains which classes the network preferred. It is not a pixel-level causal explanation; the current model does not generate a saliency or Grad-CAM map.</span></div>
        </div>
      )}
      {bestGenome && (
        <div className="below-note"><Dna size={15} /><span>Current evolved chromosome:</span><code>{`C1 ${bestGenome.c1} · C3 ${bestGenome.c2} · kernel ${bestGenome.kernel} · ${bestGenome.activation} · ${bestGenome.pooling} pool · dropout ${bestGenome.dropout}`}</code></div>
      )}
    </>
  );
}

function ExperimentLab({ dashboard }) {
  const metadata = dashboard.metadata;
  const history = dashboard.history;
  const summary = dashboard.summary;
  const convergence = history.genetic_history || [];
  const evaluations = useMemo(() => {
    const genetic = (history.genetic_evaluations || []).map((item) => ({ ...item, methodName: "Genetic Algorithm" }));
    const random = (history.random_evaluations || []).map((item) => ({ ...item, methodName: "Random Search" }));
    return [...genetic, ...random];
  }, [history]);
  const gaWinner = summary.find((item) => item.name.includes("GA-LeNet"));
  const baseline = summary.find((item) => item.name.includes("Baseline"));
  const delta = gaWinner && baseline ? (gaWinner.accuracy_mean - baseline.accuracy_mean) * 100 : 0;

  return (
    <>
      <SectionHeading eyebrow="EXPERIMENTAL EVIDENCE" title="Measure what evolution changed." description="The charts below are generated from the training run saved on this machine. The test set is held aside until final evaluation." right={<span className="mini-tag">{metadata.profile.toUpperCase()} PROFILE</span>} />
      <div className="stat-grid stat-grid-four">
        <StatCard icon={Target} label="GA test accuracy" value={gaWinner ? percent(gaWinner.accuracy_mean) : "—"} detail={`${gaWinner ? (gaWinner.accuracy_std * 100).toFixed(2) : "0.00"}% standard deviation`} tone="green" />
        <StatCard icon={Layers3} label="GA parameters" value={number(metadata.best_parameters)} detail="Trainable weights and biases" tone="cyan" />
        <StatCard icon={GitBranch} label="GA evaluations" value={metadata.genetic_unique_evaluations} detail="Unique candidate architectures" tone="violet" />
        <StatCard icon={Gauge} label="Search space" value={number(metadata.search_space_size)} detail="Possible chromosome combinations" tone="amber" />
      </div>

      <div className="chart-grid">
        <div className="panel chart-panel">
          <div className="panel-topline"><div><span className="panel-kicker">EVOLUTION TRACK</span><h3>GA fitness by generation</h3></div><span className="chart-tag">Higher is better</span></div>
          <div className="chart-area"><ResponsiveContainer width="100%" height="100%"><LineChart data={convergence} margin={{ top: 12, right: 18, left: 0, bottom: 2 }}><CartesianGrid strokeDasharray="3 3" stroke="#263247" /><XAxis dataKey="generation" stroke="#8090a8" tickLine={false} axisLine={false} /><YAxis stroke="#8090a8" tickLine={false} axisLine={false} domain={["auto", "auto"]} /><Tooltip contentStyle={{ background: "#101a2b", border: "1px solid #2b3a53", borderRadius: 10, color: "#eef4ff" }} /><Legend /><Line type="monotone" dataKey="best_fitness" name="Best fitness" stroke="#63e5cd" strokeWidth={2.5} dot={{ r: 4 }} /><Line type="monotone" dataKey="mean_fitness" name="Mean fitness" stroke="#a28bff" strokeWidth={2} strokeDasharray="5 4" dot={{ r: 3 }} /></LineChart></ResponsiveContainer></div>
          <p className="chart-caption">Elitism carries the best chromosome forward, so the best-so-far fitness should not decrease from one generation to the next.</p>
        </div>
        <div className="panel chart-panel">
          <div className="panel-topline"><div><span className="panel-kicker">MODEL COMPARISON</span><h3>Accuracy on held-out MNIST</h3></div><span className="chart-tag">Mean test accuracy</span></div>
          <div className="chart-area"><ResponsiveContainer width="100%" height="100%"><BarChart data={summary.map((item) => ({ name: item.name.replace(" (proposed)", ""), accuracy: item.accuracy_mean * 100 }))} margin={{ top: 12, right: 12, left: 0, bottom: 2 }}><CartesianGrid strokeDasharray="3 3" stroke="#263247" vertical={false} /><XAxis dataKey="name" stroke="#8090a8" tickLine={false} axisLine={false} tick={{ fontSize: 10 }} /><YAxis stroke="#8090a8" tickLine={false} axisLine={false} domain={[0, 100]} /><Tooltip contentStyle={{ background: "#101a2b", border: "1px solid #2b3a53", borderRadius: 10, color: "#eef4ff" }} formatter={(value) => [`${Number(value).toFixed(2)}%`, "Test accuracy"]} /><Bar dataKey="accuracy" radius={[6, 6, 0, 0]}>{summary.map((item, index) => <Cell key={item.name} fill={["#65738b", "#a28bff", "#63e5cd"][index]} />)}</Bar></BarChart></ResponsiveContainer></div>
          <p className="chart-caption">GA vs baseline difference: <strong className={delta >= 0 ? "positive" : "negative"}>{delta >= 0 ? "+" : ""}{delta.toFixed(2)} percentage points</strong>. A negative result is still valid evidence; do not hide it.</p>
        </div>
      </div>

      <div className="panel table-panel">
        <div className="panel-topline"><div><span className="panel-kicker">FINAL TEST RESULTS</span><h3>Three-way comparison</h3></div><span className="mini-tag">Untouched 10,000-image test split</span></div>
        <div className="table-scroll"><table><thead><tr><th>Model</th><th>Mean accuracy</th><th>Std. dev.</th><th>Parameters</th><th>Training time</th></tr></thead><tbody>{summary.map((item) => <tr key={item.name}><td><strong>{item.name}</strong>{item.name.includes("GA-LeNet") && <span className="table-badge">PROPOSED</span>}</td><td>{percent(item.accuracy_mean)}</td><td>± {(item.accuracy_std * 100).toFixed(2)}%</td><td>{number(item.parameters)}</td><td>{item.training_seconds_mean.toFixed(1)} s</td></tr>)}</tbody></table></div>
        <div className="table-footnote"><Info size={14} /> Demo profile uses one final random seed, so its standard deviation is zero by definition. Use the full profile for the three-seed mean ± standard deviation described in the report.</div>
      </div>

      <div className="figure-grid">
        {Object.entries(metadata.plots || {}).map(([key, file]) => (
          <div className="panel figure-card" key={file}><div className="figure-title"><span>{key.replaceAll("_", " ")}</span><a href={`${API}/artifacts/${file}`} target="_blank" rel="noreferrer"><ArrowUpRight size={14} /></a></div><img src={`${API}/artifacts/${file}`} alt={key.replaceAll("_", " ")} /></div>
        ))}
      </div>
      <div className="panel evaluations-panel">
        <div className="panel-topline"><div><span className="panel-kicker">SEARCH AUDIT</span><h3>Recent candidate evaluations</h3></div><span className="mini-tag">{evaluations.length} records</span></div>
        <div className="table-scroll"><table><thead><tr><th>Method</th><th>Evaluation</th><th>Generation</th><th>Validation accuracy</th><th>Fitness</th><th>Parameters</th></tr></thead><tbody>{evaluations.slice(-12).reverse().map((item, index) => <tr key={`${item.methodName}-${item.evaluation}-${index}`}><td>{item.methodName}</td><td>#{item.evaluation}</td><td>{item.generation}</td><td>{percent(item.validation_accuracy)}</td><td>{item.fitness.toFixed(4)}</td><td>{number(item.parameters)}</td></tr>)}</tbody></table></div>
      </div>
    </>
  );
}

function AlgorithmGuide({ dashboard }) {
  const metadata = dashboard.metadata;
  const settings = metadata.settings;
  const genes = Object.entries(metadata.genes);
  const searchCount = metadata.search_space_size;
  return (
    <>
      <SectionHeading eyebrow="THE ENGINE ROOM" title="A neural network, evolved by nature-inspired search." description="This is a neuro-genetic hybrid: the CNN learns from images, while the Genetic Algorithm chooses which CNN should be trained." />
      <div className="concept-grid">
        <div className="concept-card concept-neural"><div className="concept-icon"><BrainCircuit size={21} /></div><span>LEARNING ENGINE</span><h3>LeNet-style CNN</h3><p>Reads the 28 × 28 pixel image. Convolution learns visual patterns, pooling reduces the map size, dense layers combine the learned features, and softmax scores the ten digits.</p><div className="concept-foot">Learns weights using back-propagation + Adam</div></div>
        <div className="concept-card concept-genetic"><div className="concept-icon"><Dna size={21} /></div><span>DESIGN ENGINE</span><h3>Genetic Algorithm</h3><p>Creates candidate CNN designs, scores them with validation accuracy and model size, then breeds stronger candidates through selection, crossover and mutation.</p><div className="concept-foot">Searches {number(searchCount)} possible configurations</div></div>
      </div>
      <div className="panel flow-panel">
        <div className="panel-topline"><div><span className="panel-kicker">THE TRAINING CYCLE</span><h3>How the system finds a stronger design</h3></div><span className="mini-tag">Human-readable workflow</span></div>
        <div className="flow-list">
          {[
            ["01", "Create a population", "Generate a group of nine-gene chromosomes. Each chromosome describes one possible CNN."],
            ["02", "Build and train each CNN", "Translate each chromosome into layers and settings, then train it briefly on a training subset."],
            ["03", "Measure fitness", "Check validation accuracy and subtract a small penalty for a large parameter count."],
            ["04", "Select parents", "Tournament selection samples a few candidates; the fittest candidate becomes a parent."],
            ["05", "Crossover and mutate", "Uniform crossover mixes genes from two parents. Mutation randomly changes selected genes."],
            ["06", "Keep the elite", "Copy the best chromosome unchanged into the next generation, then repeat the cycle."],
            ["07", "Compare fairly", "Give random search the same number of candidate evaluations, then compare both against baseline LeNet-5."],
            ["08", "Final test", "Retrain the three finalists and evaluate them on MNIST test images that were not used during search."]
          ].map(([numberText, title, description]) => <div className="flow-item" key={numberText}><span className="flow-index">{numberText}</span><div><strong>{title}</strong><p>{description}</p></div></div>)}
        </div>
      </div>
      <div className="panel gene-panel">
        <div className="panel-topline"><div><span className="panel-kicker">CHROMOSOME DESIGN</span><h3>The nine genes the GA can change</h3></div><span className="mini-tag">Search-space definition</span></div>
        <div className="gene-grid">
          {genes.map(([gene, values], index) => <div className="gene-card" key={gene}><span>GENE {String(index + 1).padStart(2, "0")}</span><strong>{gene}</strong><p>{gene === "c1" ? "Filters in first convolution layer" : gene === "c2" ? "Filters in second convolution layer" : gene === "kernel" ? "Convolution filter width and height" : gene === "fc1" ? "Neurons in first dense layer" : gene === "fc2" ? "Neurons in second dense layer" : gene === "activation" ? "Non-linear activation function" : gene === "pooling" ? "Spatial downsampling method" : gene === "dropout" ? "Fraction of dense activations dropped during training" : "Adam optimiser learning rate"}</p><div className="gene-values">{values.map((value) => <code key={String(value)}>{String(value)}</code>)}</div></div>)}
        </div>
        <div className="search-total"><div><span>Possible networks</span><strong>{number(searchCount)}</strong></div><div><span>Genes per chromosome</span><strong>9</strong></div><div><span>Current best fitness</span><strong>{metadata.best_fitness.toFixed(4)}</strong></div></div>
      </div>
      <div className="two-column-info">
        <div className="panel info-panel">
          <div className="panel-topline"><div><span className="panel-kicker">FITNESS FUNCTION</span><h3>Accuracy with a size penalty</h3></div><Gauge size={20} className="muted-icon" /></div>
          <div className="formula">Fitness = Validation Accuracy − λ × (Parameters / 100,000)</div>
          <p>Validation accuracy rewards a correct classifier. The size penalty discourages a huge network when a smaller network performs almost as well. In this implementation, λ = {settings.lambda_size}.</p>
          <div className="parameter-pair"><span>Higher accuracy</span><b>+</b><span>Fewer parameters</span><b>=</b><strong>Better fitness</strong></div>
        </div>
        <div className="panel info-panel">
          <div className="panel-topline"><div><span className="panel-kicker">NEURAL LEARNING</span><h3>How weights get updated</h3></div><Zap size={20} className="muted-icon" /></div>
          <p>The Genetic Algorithm does not adjust the millions of image pixels directly. It chooses the CNN settings. Inside each chosen CNN, back-propagation calculates how the prediction error changes with each weight, and Adam updates those weights to reduce the loss.</p>
          <div className="learning-equation">Image → CNN → loss → back-propagation → Adam update</div>
        </div>
      </div>
      <div className="panel param-panel">
        <div className="panel-topline"><div><span className="panel-kicker">ACTUAL TRAINING PARAMETERS</span><h3>Settings used in this saved run</h3></div><span className="mini-tag">From run metadata</span></div>
        <div className="parameter-grid">
          {[
            ["Population size", settings.population], ["Generations", settings.generations], ["Training epochs per candidate", settings.fit_epochs], ["GA training images", number(metadata.dataset_split.ga_training_subset)], ["GA validation images", number(metadata.dataset_split.ga_validation_subset)], ["Crossover probability", settings.crossover_probability], ["Per-gene mutation probability", settings.mutation_probability], ["Tournament size", settings.tournament_size], ["Elitism", `${settings.elite} chromosome`], ["Size penalty λ", settings.lambda_size], ["Final training epochs", settings.final_epochs], ["Final random seeds", settings.final_runs]
          ].map(([label, value]) => <div className="parameter-item" key={label}><span>{label}</span><strong>{value}</strong></div>)}
        </div>
      </div>
    </>
  );
}

function ResearchNotes({ dashboard }) {
  const metadata = dashboard.metadata;
  const split = metadata.dataset_split;
  const genes = metadata.genes;
  return (
    <>
      <SectionHeading eyebrow="PROJECT STORY & TESTING" title="What is original, what is upgraded, and how it is tested." description="Use this page to explain the project to your professor: research foundation, your contribution, data source, and the rules used to keep the comparison fair." />
      <div className="paper-card">
        <div className="paper-icon"><BookOpen size={24} /></div>
        <div className="paper-copy"><span className="panel-kicker">BASE RESEARCH PAPER · 1998</span><h3>Gradient-Based Learning Applied to Document Recognition</h3><p>Yann LeCun, Léon Bottou, Yoshua Bengio and Patrick Haffner · Proceedings of the IEEE, Volume 86, Issue 11, pages 2278–2324 (1998).</p><a href="https://doi.org/10.1109/5.726791" target="_blank" rel="noreferrer">Open the research paper by DOI <ArrowUpRight size={14} /></a></div>
        <div className="paper-year">1998</div>
      </div>
      <div className="panel novelty-panel">
        <div className="panel-topline"><div><span className="panel-kicker">OUR CONTRIBUTION</span><h3>What we added on top of the paper</h3></div><span className="contribution-tag">PROJECT UPGRADE</span></div>
        <div className="novelty-list">
          <div className="novelty-item"><div className="novelty-no">01</div><div><strong>Automatic architecture search</strong><p>The paper presents the LeNet family of CNNs. This project adds a Genetic Algorithm that searches filters, kernels, dense sizes, activation, pooling, dropout and learning rate instead of fixing all of them by hand.</p></div></div>
          <div className="novelty-item"><div className="novelty-no">02</div><div><strong>Compactness-aware fitness</strong><p>Our fitness function rewards validation accuracy but also penalises extra trainable parameters. This makes model size part of the design objective.</p></div></div>
          <div className="novelty-item"><div className="novelty-no">03</div><div><strong>Fair baseline comparison</strong><p>The original LeNet-style settings are included as a valid chromosome. GA-LeNet and random search are evaluated with the same candidate-evaluation budget.</p></div></div>
          <div className="novelty-item"><div className="novelty-no">04</div><div><strong>Modern implementation and analysis</strong><p>This re-implementation uses Adam, softmax with cross-entropy, a fully connected second convolution block, and optional ReLU, max pooling and dropout. These are implementation choices, not claims that the original paper used them.</p></div></div>
          <div className="novelty-item"><div className="novelty-no">05</div><div><strong>Separate validation and test data</strong><p>The GA sees the validation split only. The untouched test split is reserved for the final comparison to reduce test-set leakage.</p></div></div>
        </div>
      </div>
      <div className="two-column-info">
        <div className="panel info-panel">
          <div className="panel-topline"><div><span className="panel-kicker">INPUT DATASET</span><h3>Where the inputs come from</h3></div><Database size={20} className="muted-icon" /></div>
          <p>Training uses MNIST: 70,000 grayscale handwritten digit images, each 28 × 28 pixels. Keras downloads it automatically the first time the training script runs.</p>
          <div className="dataset-counts"><div><strong>60,000</strong><span>Training pool</span></div><div><strong>10,000</strong><span>Test images</span></div><div><strong>0–9</strong><span>Digit classes</span></div></div>
          <a className="text-link" href="https://keras.io/api/datasets/mnist/" target="_blank" rel="noreferrer">MNIST dataset documentation <ArrowUpRight size={13} /></a>
          <a className="text-link" href="https://www.tensorflow.org/datasets/catalog/mnist" target="_blank" rel="noreferrer">TensorFlow Datasets: MNIST <ArrowUpRight size={13} /></a>
        </div>
        <div className="panel info-panel">
          <div className="panel-topline"><div><span className="panel-kicker">LIVE DEMO INPUT</span><h3>What should I give it?</h3></div><FileImage size={20} className="muted-icon" /></div>
          <p>For the classroom demonstration, use a single handwritten digit, not a full sentence. You can draw it in Live Studio, upload a cropped PNG/JPG, or click a built-in MNIST example with a known label.</p>
          <ul className="simple-list"><li>Draw a clear 0, 1, 2, … 9 on the black canvas.</li><li>Upload an image with one centered dark or light digit.</li><li>Choose a built-in sample to show a known test example.</li></ul>
        </div>
      </div>
      <div className="panel testing-panel">
        <div className="panel-topline"><div><span className="panel-kicker">TESTING PARAMETERS</span><h3>How this experiment was checked</h3></div><ShieldCheckIcon /></div>
        <div className="split-bar"><div style={{ width: `${split.training / (split.training + split.validation) * 100}%` }} /><div style={{ width: `${split.validation / (split.training + split.validation) * 100}%` }} /></div>
        <div className="split-legend"><span><i className="dot train-dot" /> Training · {number(split.training)}</span><span><i className="dot val-dot" /> Validation · {number(split.validation)}</span><span><i className="dot test-dot" /> Test · {number(split.test)}</span></div>
        <div className="testing-grid">
          <div><span>Candidate evaluation subset</span><strong>{number(split.ga_training_subset)} train + {number(split.ga_validation_subset)} validation</strong><p>Small subset keeps each candidate evaluation practical.</p></div>
          <div><span>Final test set</span><strong>{number(split.test)} held-out images</strong><p>Used only after GA and random search have selected their winners.</p></div>
          <div><span>Equal-budget control</span><strong>{metadata.genetic_unique_evaluations} vs {metadata.random_unique_evaluations}</strong><p>Genetic and random search receive the same number of unique candidate evaluations.</p></div>
          <div><span>Hardware detected</span><strong>{metadata.hardware.compute_device}</strong><p>{metadata.hardware.gpu_devices?.length ? metadata.hardware.gpu_devices.join(", ") : "TensorFlow did not detect a supported GPU; training runs on CPU."}</p></div>
        </div>
        <div className="honesty-note"><Info size={15} /><span>The app reports the compute device, parameter count, estimated multiply-accumulate operations and measured inference time. It does not measure electrical power in watts.</span></div>
      </div>
      <div className="panel report-note">
        <div className="report-note-icon"><BookOpen size={19} /></div>
        <div><strong>Important for your report</strong><p>Use the actual values shown in Experiment Lab. The demo profile uses one seed and fewer epochs. For the multi-seed results described in the synopsis/report, run <code>python train.py --profile full</code>, then use the newly generated results. Never fill the report with expected or invented accuracy values.</p></div>
      </div>
    </>
  );
}

function ShieldCheckIcon() {
  return <CheckCircle2 size={20} className="muted-icon" />;
}

function App() {
  const [tab, setTab] = useState("studio");
  const [status, setStatus] = useState({ status: "not_started", stage: "Not trained yet", percent: 0, logs: [] });
  const [dashboard, setDashboard] = useState(null);
  const [examples, setExamples] = useState([]);
  const [prediction, setPrediction] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [serverOnline, setServerOnline] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const health = await fetch(`${API}/api/health`);
      setServerOnline(health.ok);
      if (!health.ok) return;
      const statusResponse = await fetch(`${API}/api/status`);
      if (statusResponse.ok) setStatus(await statusResponse.json());
      const dashboardResponse = await fetch(`${API}/api/dashboard`);
      if (dashboardResponse.ok) {
        const data = await dashboardResponse.json();
        setDashboard(data);
        if (!examples.length) {
          const examplesResponse = await fetch(`${API}/api/examples`);
          if (examplesResponse.ok) {
            const sampleData = await examplesResponse.json();
            setExamples(sampleData.examples || []);
          }
        }
      }
    } catch {
      setServerOnline(false);
    }
  }, [examples.length]);

  useEffect(() => {
    refresh();
    const timer = window.setInterval(refresh, 3500);
    return () => window.clearInterval(timer);
  }, [refresh]);

  const onPredict = async (file) => {
    setBusy(true);
    setError("");
    setPrediction(null);
    const form = new FormData();
    form.append("file", file);
    try {
      const response = await fetch(`${API}/api/predict`, { method: "POST", body: form });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || "Prediction failed.");
      setPrediction(result);
    } catch (failure) {
      setError(failure.message || "Could not reach the model API. Check that the backend is running.");
    } finally {
      setBusy(false);
    }
  };

  const currentIndex = navItems.findIndex((item) => item.id === tab);
  const modelReady = Boolean(dashboard && status.model_ready);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-lockup"><div className="brand-mark"><Dna size={22} /></div><div><strong>NEURO<span>GENETIC</span></strong><small>SOFT COMPUTING LAB</small></div></div>
        <div className="sidebar-label">WORKSPACE</div>
        <nav className="side-nav">
          {navItems.map((item) => {
            const Icon = item.icon;
            return <button key={item.id} className={tab === item.id ? "active" : ""} onClick={() => setTab(item.id)}><Icon size={17} /><span>{item.label}</span>{tab === item.id && <ChevronRight size={14} />}</button>;
          })}
        </nav>
        <div className="sidebar-bottom">
          <div className="sidebar-project"><div className="project-icon"><BrainCircuit size={17} /></div><div><strong>LeNet-5 + GA</strong><span>Hybrid intelligence</span></div></div>
          <div className="sidebar-status"><span className={modelReady ? "status-dot online" : "status-dot"} /><span>{modelReady ? "Model ready" : status.status === "running" ? "Training in progress" : "Waiting for training"}</span></div>
          <div className="sidebar-footer">BUILT FOR SOFT COMPUTING<br />ACADEMIC PROJECT · 2026–27</div>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="breadcrumb"><span>PROJECTS</span><ChevronRight size={13} /><strong>NEURO-GENETIC LENET-5</strong></div>
          <div className="topbar-right"><div className={`connection ${serverOnline ? "connected" : ""}`}><span />{serverOnline ? "API CONNECTED" : "API OFFLINE"}</div><div className={`model-state ${modelReady ? "ready" : "not-ready"}`}>{modelReady ? <CheckCircle2 size={14} /> : <CircleHelp size={14} />}{modelReady ? "MODEL READY" : "MODEL NOT TRAINED"}</div></div>
        </header>

        <div className="content-area">
          <div className="hero-row">
            <div><div className="hero-eyebrow"><span className="hero-line" /> INTELLIGENT EXPERIMENTATION PLATFORM</div><h1>Neuro-Genetic <em>LeNet-5</em></h1><p>A CNN that learns digits. A Genetic Algorithm that evolves the brain behind it.</p></div>
            <div className="hero-orb"><div className="orb-ring ring-one" /><div className="orb-ring ring-two" /><div className="orb-core"><BrainCircuit size={30} /></div><span className="orb-node node-one" /><span className="orb-node node-two" /><span className="orb-node node-three" /></div>
          </div>

          {!serverOnline && <div className="backend-offline"><AlertCircle size={16} /><span>Backend is not connected. Start it in VS Code with <code>python -m uvicorn backend.api:app --reload</code>.</span></div>}
          <TrainingBanner status={status} />

          <div className="page-tabs">
            {navItems.map((item, index) => {
              const Icon = item.icon;
              return <button key={item.id} className={tab === item.id ? "active" : ""} onClick={() => setTab(item.id)}><Icon size={15} /><span>{item.label}</span><small>0{index + 1}</small></button>;
            })}
          </div>

          {!modelReady ? <EmptyTraining status={status} onRefresh={refresh} /> : (
            <>
              {tab === "studio" && <LiveStudio dashboard={dashboard} examples={examples} prediction={prediction} setPrediction={setPrediction} onPredict={onPredict} busy={busy} error={error} setError={setError} />}
              {tab === "experiment" && <ExperimentLab dashboard={dashboard} />}
              {tab === "algorithm" && <AlgorithmGuide dashboard={dashboard} />}
              {tab === "research" && <ResearchNotes dashboard={dashboard} />}
            </>
          )}

          <footer className="page-footer"><div><span className="footer-mark"><Dna size={14} /></span><strong>NEUROGENETIC LAB</strong><span>·</span><span>Neural learning meets evolutionary search</span></div><span>MNIST · LENET-5 · GENETIC ALGORITHM</span></footer>
        </div>
      </main>
    </div>
  );
}

export default App;
