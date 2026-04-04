const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, VerticalAlign, PageNumber, PageBreak, LevelFormat,
  ImageRun
} = require("docx");
const fs = require("fs");

// ─── Images ───────────────────────────────────────────────────────────────────
const IMG_ENV   = fs.readFileSync("results/diagrams/environment_diagram.png");
const IMG_CURVES = fs.readFileSync("C:/tmp/nb_img_cell15_out0.png");
const IMG_BAR    = fs.readFileSync("C:/tmp/nb_img_cell16_out0.png");

// Image helper (EMU: 1 inch = 914400)
function img(data, widthEMU, heightEMU, title) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 60 },
    children: [new ImageRun({
      type: "png",
      data,
      transformation: { width: Math.round(widthEMU / 9144), height: Math.round(heightEMU / 9144) },
      altText: { title, description: title, name: title },
    })],
  });
}

// ─── Constants ────────────────────────────────────────────────────────────────
const FONT = "Arial";
const PAGE_W = 11906; // A4 width DXA
const PAGE_H = 16838; // A4 height DXA
const MARGIN = 1440;  // 1 inch
const CONTENT_W = PAGE_W - MARGIN * 2; // 9026

// ─── Colour palette ───────────────────────────────────────────────────────────
const BLUE       = "2E5DAD";
const DARK_GRAY  = "434343";
const MED_GRAY   = "666666";
const LIGHT_BLUE = "D5E3F5";
const HEADER_BG  = "2E5DAD";
const ALT_ROW    = "EEF3FB";
const WHITE      = "FFFFFF";

// ─── Border helpers ───────────────────────────────────────────────────────────
const thin  = { style: BorderStyle.SINGLE, size: 4,  color: "BBBBBB" };
const thick = { style: BorderStyle.SINGLE, size: 8,  color: BLUE };
const none  = { style: BorderStyle.NONE,   size: 0,  color: "FFFFFF" };
const cellBorders = { top: thin, bottom: thin, left: thin, right: thin };
const headerBorders = { top: thick, bottom: thick, left: thin, right: thin };

// ─── Helper: plain paragraph ──────────────────────────────────────────────────
function p(text, opts = {}) {
  return new Paragraph({
    spacing: { before: 60, after: 100 },
    children: [new TextRun({ text, font: FONT, size: 22, ...opts })],
    alignment: opts.align || AlignmentType.JUSTIFIED,
  });
}

function pEmpty() {
  return new Paragraph({ children: [new TextRun("")], spacing: { before: 40, after: 40 } });
}

// ─── Helper: heading ──────────────────────────────────────────────────────────
function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 160 },
    children: [new TextRun({ text, font: FONT, size: 40, bold: true, color: BLUE })],
  });
}

function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 280, after: 120 },
    children: [new TextRun({ text, font: FONT, size: 32, bold: true, color: "000000" })],
    border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: BLUE, space: 2 } },
  });
}

function h3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 200, after: 80 },
    children: [new TextRun({ text, font: FONT, size: 28, bold: true, color: DARK_GRAY })],
  });
}

function h4(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_4,
    spacing: { before: 140, after: 60 },
    children: [new TextRun({ text, font: FONT, size: 24, bold: false, color: MED_GRAY })],
  });
}

// ─── Helper: bullet ───────────────────────────────────────────────────────────
function bullet(text, bold_prefix = "") {
  const children = [];
  if (bold_prefix) {
    children.push(new TextRun({ text: bold_prefix, font: FONT, size: 22, bold: true }));
  }
  children.push(new TextRun({ text, font: FONT, size: 22 }));
  return new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    spacing: { before: 40, after: 40 },
    children,
  });
}

// ─── Helper: bold label + text ────────────────────────────────────────────────
function labelPara(label, text) {
  return new Paragraph({
    spacing: { before: 60, after: 80 },
    children: [
      new TextRun({ text: label + ": ", font: FONT, size: 22, bold: true }),
      new TextRun({ text, font: FONT, size: 22 }),
    ],
    alignment: AlignmentType.JUSTIFIED,
  });
}

// ─── Helper: table header cell ────────────────────────────────────────────────
function thCell(text, w) {
  return new TableCell({
    borders: headerBorders,
    width: { size: w, type: WidthType.DXA },
    shading: { fill: HEADER_BG, type: ShadingType.CLEAR },
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ text, font: FONT, size: 18, bold: true, color: WHITE })],
    })],
  });
}

// ─── Helper: table data cell ──────────────────────────────────────────────────
function tdCell(text, w, isAlt = false, center = false) {
  return new TableCell({
    borders: cellBorders,
    width: { size: w, type: WidthType.DXA },
    shading: { fill: isAlt ? ALT_ROW : WHITE, type: ShadingType.CLEAR },
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({
      alignment: center ? AlignmentType.CENTER : AlignmentType.LEFT,
      children: [new TextRun({ text, font: FONT, size: 20 })],
    })],
  });
}

// ─── Helper: bold-name data cell ──────────────────────────────────────────────
function tdBold(text, w, isAlt = false) {
  return new TableCell({
    borders: cellBorders,
    width: { size: w, type: WidthType.DXA },
    shading: { fill: isAlt ? ALT_ROW : WHITE, type: ShadingType.CLEAR },
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({
      children: [new TextRun({ text, font: FONT, size: 20, bold: true })],
    })],
  });
}

// ─── 2-column info table (label | value) ─────────────────────────────────────
function infoTable(rows) {
  const colA = Math.round(CONTENT_W * 0.35);
  const colB = CONTENT_W - colA;
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: [colA, colB],
    rows: rows.map(([label, val], i) => new TableRow({
      children: [
        new TableCell({
          borders: cellBorders,
          width: { size: colA, type: WidthType.DXA },
          shading: { fill: i % 2 === 0 ? LIGHT_BLUE : WHITE, type: ShadingType.CLEAR },
          margins: { top: 60, bottom: 60, left: 120, right: 120 },
          children: [new Paragraph({ children: [new TextRun({ text: label, font: FONT, size: 20, bold: true })] })],
        }),
        new TableCell({
          borders: cellBorders,
          width: { size: colB, type: WidthType.DXA },
          shading: { fill: i % 2 === 0 ? LIGHT_BLUE : WHITE, type: ShadingType.CLEAR },
          margins: { top: 60, bottom: 60, left: 120, right: 120 },
          children: [new Paragraph({ children: [new TextRun({ text: val, font: FONT, size: 20 })] })],
        }),
      ],
    })),
  });
}

// ─── Generic data table ───────────────────────────────────────────────────────
function dataTable(headers, rows, colWidths) {
  const totalW = colWidths.reduce((a, b) => a + b, 0);
  const headerRow = new TableRow({
    tableHeader: true,
    children: headers.map((h, i) => thCell(h, colWidths[i])),
  });
  const dataRows = rows.map((row, ri) => new TableRow({
    children: row.map((cell, ci) =>
      ci === 0
        ? tdBold(cell, colWidths[ci], ri % 2 !== 0)
        : tdCell(cell, colWidths[ci], ri % 2 !== 0, true)
    ),
  }));
  return new Table({
    width: { size: totalW, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [headerRow, ...dataRows],
  });
}

// ─── Caption paragraph ────────────────────────────────────────────────────────
function caption(text) {
  return new Paragraph({
    spacing: { before: 60, after: 160 },
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text, font: FONT, size: 18, italics: true, color: MED_GRAY })],
  });
}

// ─── Page break ───────────────────────────────────────────────────────────────
function pb() { return new Paragraph({ children: [new PageBreak()] }); }

// ═══════════════════════════════════════════════════════════════════════════════
//  DATA
// ═══════════════════════════════════════════════════════════════════════════════

const dqnHeaders  = ["Run Name", "Learning Rate", "Gamma", "Replay Buffer", "Batch Size", "Exploration (start\u2192end)", "Final Mean Reward"];
const dqnColW     = [1800, 1100, 800, 1100, 900, 1500, 1026];
const dqnRows = [
  ["dqn_large_buffer",   "1e-4",  "0.995", "300,000", "128", "0.65 \u2192 0.02", "\u221250.29"],
  ["dqn_low_lr",         "1e-5",  "0.995", "100,000", "64",  "0.60 \u2192 0.05", "\u221259.80"],
  ["dqn_low_gamma",      "1e-4",  "0.950", "100,000", "64",  "0.65 \u2192 0.05", "\u221267.37"],
  ["dqn_high_lr",        "5e-4",  "0.995", "100,000", "64",  "0.60 \u2192 0.05", "\u221267.58"],
  ["dqn_med_lr_batch",   "3e-4",  "0.995", "200,000", "256", "0.60 \u2192 0.05", "\u221269.83"],
  ["dqn_small_net",      "1e-4",  "0.995", "100,000", "64",  "0.60 \u2192 0.05", "\u221274.21"],
  ["dqn_deep_net",       "1e-4",  "0.995", "100,000", "64",  "0.60 \u2192 0.05", "\u221275.59"],
  ["dqn_long_explore",   "1e-4",  "0.995", "100,000", "64",  "0.75 \u2192 0.01", "\u221279.07"],
  ["dqn_soft_update",    "1e-4",  "0.995", "100,000", "64",  "0.60 \u2192 0.05", "\u221281.29"],
  ["dqn_baseline",       "1e-4",  "0.995", "100,000", "64",  "0.60 \u2192 0.05", "\u221287.89"],
];

const rfHeaders  = ["Run Name", "Learning Rate", "Gamma", "Hidden Layers", "Baseline Strategy", "Final Mean Reward"];
const rfColW     = [1900, 1100, 800, 1300, 1500, 1426];
const rfRows = [
  ["reinforce_deep_net",        "1e-3",  "0.99", "[256, 128, 64]", "Mean (episode)",   "+1319.85"],
  ["reinforce_large_net",       "1e-3",  "0.99", "[256, 128]",     "Mean (episode)",   "+752.62"],
  ["reinforce_very_low_gamma",  "1e-3",  "0.90", "[128, 64]",      "Mean (episode)",   "+493.00"],
  ["reinforce_running_baseline","3e-4",  "0.99", "[128, 64]",      "Running EMA",      "+475.77"],
  ["reinforce_low_gamma",       "1e-3",  "0.95", "[128, 64]",      "Mean (episode)",   "+448.98"],
  ["reinforce_no_baseline",     "1e-3",  "0.99", "[128, 64]",      "None (vanilla)",   "+391.09"],
  ["reinforce_small_net",       "1e-3",  "0.99", "[64, 32]",       "Mean (episode)",   "+209.63"],
  ["reinforce_baseline",        "1e-3",  "0.99", "[128, 64]",      "Mean (episode)",   "+202.09"],
  ["reinforce_low_lr",          "1e-4",  "0.99", "[128, 64]",      "Mean (episode)",   "+115.11"],
  ["reinforce_high_lr",         "5e-3",  "0.99", "[128, 64]",      "Mean (episode)",   "+65.02"],
];

const ppoHeaders = ["Run Name", "Learning Rate", "Clip \u03B5", "Entropy Coef", "n_steps", "Batch Size", "Final Mean Reward"];
const ppoColW    = [1700, 1100, 900, 1100, 900, 1000, 1026];
const ppoRows = [
  ["ppo_tight_clip",     "3e-4",   "0.15", "0.01", "2048", "128", "+364.73"],
  ["ppo_wide_clip",      "3e-4",   "0.30", "0.01", "2048", "128", "+353.83"],
  ["ppo_baseline",       "3e-4",   "0.20", "0.01", "2048", "64",  "+345.87"],
  ["ppo_low_gamma",      "3e-4",   "0.20", "0.01", "2048", "64",  "+337.59"],
  ["ppo_large_rollout",  "3e-4",   "0.20", "0.01", "4096", "256", "+325.50"],
  ["ppo_low_lr",         "1.5e-4", "0.20", "0.01", "2048", "64",  "+318.97"],
  ["ppo_small_net",      "3e-4",   "0.20", "0.01", "2048", "64",  "+253.83"],
  ["ppo_high_lr_deep",   "5e-4",   "0.20", "0.01", "2048", "64",  "+253.76"],
  ["ppo_more_epochs",    "3e-4",   "0.20", "0.01", "2048", "64",  "+214.30"],
  ["ppo_high_entropy",   "3e-4",   "0.20", "0.05", "2048", "64",  "+184.03"],
];

// Observation space rows
const obsRows = [
  ["[0\u20132]",   "Spacecraft position (x, y, z)",                "AU",    "[\u221250, +50]"],
  ["[3\u20135]",   "Spacecraft velocity (vx, vy, vz)",             "AU/s",  "[\u221210, +10]"],
  ["[6]",          "Normalised distance to Mars",                   "\u2014", "[0, 1]"],
  ["[7\u20139]",   "Unit heading vector to Mars",                   "\u2014", "[\u22121, 1]\u00B3"],
  ["[10]",         "Normalised distance to Europa",                 "\u2014", "[0, 1]"],
  ["[11\u201313]", "Unit heading vector to Europa",                 "\u2014", "[\u22121, 1]\u00B3"],
  ["[14]",         "Normalised distance to Enceladus",              "\u2014", "[0, 1]"],
  ["[15\u201317]", "Unit heading vector to Enceladus",              "\u2014", "[\u22121, 1]\u00B3"],
  ["[18]",         "Fuel level",                                    "\u2014", "[0, 1]"],
  ["[19]",         "Battery level",                                 "\u2014", "[0, 1]"],
  ["[20]",         "Biosignature SNR signal",                       "\u2014", "[0, 1]"],
  ["[21]",         "Biosignatures found / 3",                       "\u2014", "[0, 1]"],
  ["[22]",         "Biosignatures transmitted / 3",                 "\u2014", "[0, 1]"],
  ["[23]",         "Spacecraft yaw / \u03C0",                       "\u2014", "[\u22121, 1]"],
  ["[24]",         "Normalised distance to Sun",                    "\u2014", "[0, 1]"],
  ["[25]",         "Active instrument index / 3",                   "\u2014", "[0, 1]"],
];

const obsHeaders = ["Index", "Feature", "Unit", "Range"];
const obsColW    = [800, 4000, 1000, 3226];

// Action space rows
const actRows = [
  ["0", "Thrust",      "5", "{0, 0.25, 0.5, 0.75, 1.0} normalised; max 0.001 AU/s\u00B2"],
  ["1", "Pitch",       "3", "{\u22125\u00B0, 0\u00B0, +5\u00B0} attitude adjustment"],
  ["2", "Yaw",         "3", "{\u22125\u00B0, 0\u00B0, +5\u00B0} attitude adjustment"],
  ["3", "Instrument",  "4", "{None, Spectrometer, ThermalImager, Drill}"],
  ["4", "Comms",       "2", "{Off, Transmit} biosignature data"],
];
const actHeaders = ["Dim", "Name", "# Values", "Description"];
const actColW    = [700, 1400, 900, 6026];

// Reward table rows
const rwdRows = [
  ["Approach reward",        "max(0, \u0394d) \u00D7 5.0",      "Continuous",  "Progress toward target body"],
  ["Heading alignment",      "max(0, dot) \u00D7 1.5",          "Continuous",  "Nose pointing toward target"],
  ["Liquid water detected",  "+500",                             "Sparse",      "High-value astrobiology find"],
  ["Ice detected",           "+300",                             "Sparse",      "Secondary biosig marker"],
  ["Organic compounds",      "+750",                             "Sparse",      "Key life indicator"],
  ["Signs of intelligence",  "+5,000",                           "Sparse",      "Terminal jackpot reward"],
  ["Biosig transmission",    "+200 per transmission",            "Sparse",      "Goal: transmit data to Earth"],
  ["Orbital insertion",      "+100",                             "Sparse",      "Bonus for stable orbit"],
  ["Fuel penalty",           "\u22120.001 \u00D7 fuel used",    "Dense",       "Encourages fuel efficiency"],
  ["Time penalty",           "\u22120.0001 / step",             "Dense",       "Encourages task completion"],
  ["Collision",              "\u2212100",                        "Terminal",    "Episode ends"],
  ["Out of bounds (>50 AU)", "\u2212100",                        "Terminal",    "Episode ends"],
];
const rwdHeaders = ["Component", "Value", "Type", "Effect"];
const rwdColW    = [2200, 1800, 1200, 3826];

// ═══════════════════════════════════════════════════════════════════════════════
//  BUILD DOCUMENT
// ═══════════════════════════════════════════════════════════════════════════════

const children = [];

// ── TITLE BLOCK ───────────────────────────────────────────────────────────────
children.push(
  pEmpty(),
  new Paragraph({
    spacing: { before: 600, after: 200 },
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Reinforcement Learning Summative Assignment Report", font: FONT, size: 52, bold: true, color: BLUE })],
  }),
  new Paragraph({
    spacing: { before: 0, after: 80 },
    alignment: AlignmentType.CENTER,
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: BLUE, space: 4 } },
    children: [new TextRun({ text: "AstroLogic \u2014 Deep-Space Biosignature Exploration", font: FONT, size: 30, italics: true, color: DARK_GRAY })],
  }),
  pEmpty(),
  infoTable([
    ["Student Name",       "[Your Name]"],
    ["Course",             "Machine Learning Techniques II"],
    ["Video Recording",    "[Link to your 3-min video \u2014 camera on, share entire screen]"],
    ["GitHub Repository",  "[Link to your repository]"],
    ["Date",               "April 2026"],
  ]),
  pb(),
);

// ── 1. PROJECT OVERVIEW ───────────────────────────────────────────────────────
children.push(
  h1("1. Project Overview"),
  p(
    "AstroLogic is a custom OpenAI Gymnasium environment that simulates an autonomous spacecraft exploring " +
    "a physically-inspired, scaled solar system to detect and transmit biosignatures from three target bodies: " +
    "Mars, Europa (a moon of Jupiter), and Enceladus (a moon of Saturn). The core research problem addressed " +
    "by this project is how to train a reinforcement learning agent to navigate a large, continuous, " +
    "multi-objective 3D space whilst managing constrained resources (fuel and battery) and sparse, delayed " +
    "rewards. Three RL algorithms are benchmarked \u2014 Deep Q-Network (DQN), Proximal Policy Optimisation " +
    "(PPO), and REINFORCE \u2014 each with ten hyperparameter configurations, yielding thirty systematic " +
    "experiments. The environment is implemented in Python using stable-baselines3, PyTorch, and a custom " +
    "pygame 2D renderer, with APIs designed for easy serialisation to JSON and integration into a web or " +
    "mobile front-end. The overarching aim is to evaluate exploration\u2013exploitation trade-offs, convergence " +
    "behaviour, and the practical effectiveness of policy-gradient versus value-based methods in a rich, " +
    "scientifically-motivated control task."
  ),
);

// ── 2. ENVIRONMENT DESCRIPTION ────────────────────────────────────────────────
children.push(
  h1("2. Environment Description"),

  h2("2.1 Agent"),
  p(
    "The agent is an autonomous spacecraft probe placed at a random initial position within 5 AU of the Sun. " +
    "It must independently plan trajectories, manage thrust and attitude, deploy scientific instruments, and " +
    "transmit collected biosignature data back to Earth. The spacecraft carries four instruments " +
    "(Spectrometer, Thermal Imager, Drill, and a baseline idle mode), consumes fuel proportionally to thrust " +
    "applied, and recharges its battery via solar panels when within 1.5 AU of the Sun. The agent\u2019s " +
    "learned policy must balance long-range navigation efficiency against instrument operation timing and " +
    "communication windows."
  ),

  // Environment diagram: 2812x2041px → scale to content width 6.27in = 5,734,000 EMU
  img(IMG_ENV, 5734000, Math.round(2041/2812*5734000), "AstroLogic Environment Diagram"),
  caption("Figure 1 \u2014 AstroLogic environment: scaled solar system with spacecraft, target bodies, and biosignature zones"),
  pEmpty(),

  h2("2.2 Action Space"),
  p(
    "The environment exposes a MultiDiscrete([5, 3, 3, 4, 2]) action space comprising 360 unique " +
    "joint actions across five independent sub-dimensions. DQN receives a flattened Discrete(360) " +
    "representation via a custom FlattenMultiDiscreteToDiscrete wrapper, while PPO and REINFORCE operate " +
    "directly on the native MultiDiscrete space."
  ),
  pEmpty(),
  dataTable(actHeaders, actRows, actColW),
  caption("Table 1 \u2014 Action Space: 5 \u00D7 3 \u00D7 3 \u00D7 4 \u00D7 2 = 360 discrete joint actions"),
  pEmpty(),

  h2("2.3 Observation Space"),
  p(
    "The observation is a 26-dimensional continuous vector encoding the spacecraft\u2019s kinematic state, " +
    "resource levels, scientific progress, and relative positions of all target bodies. All elements are " +
    "normalised to aid neural network training."
  ),
  pEmpty(),
  dataTable(obsHeaders, obsRows, obsColW),
  caption("Table 2 \u2014 26-Dimensional Observation Vector"),
  pEmpty(),

  h2("2.4 Reward Structure"),
  p(
    "The reward function R(s, a, s\u2032) combines dense shaping signals with sparse goal rewards and " +
    "terminal penalties. Dense components guide the agent toward targets every step; sparse components " +
    "deliver large payoffs for scientific discoveries and data transmission; terminal penalties enforce " +
    "safety constraints. The mixture creates a challenging credit-assignment problem typical of real " +
    "mission planning."
  ),
  pEmpty(),
  dataTable(rwdHeaders, rwdRows, rwdColW),
  caption("Table 3 \u2014 Reward Function Components"),
  pEmpty(),

  h2("2.5 Termination Conditions"),
  p("An episode ends under any of the following conditions:"),
  bullet("Collision: spacecraft distance < 10\u00D7 body radius (minimum 0.001 AU) \u2014 penalty \u2212100"),
  bullet("Out of Bounds: spacecraft position > 50 AU from origin \u2014 penalty \u2212100"),
  bullet("Resource Depletion: fuel \u2264 0 or battery \u2264 0"),
  bullet("Mission Success: 3 or more biosignatures transmitted to Earth"),
  bullet("Timeout: maximum 10,000 timesteps reached (episode truncation)"),
  p(
    "The celestial system comprises the Sun (fixed), Mars, Jupiter, and Saturn (heliocentric orbits), " +
    "plus Europa (Jovian orbit) and Enceladus (Saturnian orbit). Gravitational dynamics are modelled " +
    "using Newtonian gravity with a normalised gravitational constant G = 4\u03C0\u00B2/365.25\u00B2."
  ),
);

// ── 3. SYSTEM ANALYSIS AND DESIGN ────────────────────────────────────────────
children.push(
  h1("3. System Analysis and Design"),

  h2("3.1 Deep Q-Network (DQN)"),
  p(
    "DQN approximates the optimal action-value function Q*(s, a) using a multilayer perceptron with " +
    "two hidden layers of 256 neurons each (ReLU activations), implemented via stable-baselines3. " +
    "Because the environment\u2019s MultiDiscrete action space must be collapsed to a single Discrete " +
    "dimension, a custom FlattenMultiDiscreteToDiscrete wrapper converts the 5-tuple action into a " +
    "single index in {0, \u2026, 359}."
  ),
  p(
    "Key design choices: (1) Prioritised Experience Replay (PER) is enabled across all runs to improve " +
    "sample efficiency for the rare high-reward biosignature transitions. PER samples transitions with " +
    "probability proportional to |TD error|\u1D43 (\u03B1=0.6) with importance-sampling correction " +
    "(\u03B2=0.4). (2) A target network is maintained and either hard-copied (\u03C4=1.0, default) or " +
    "soft-updated (\u03C4=0.005, dqn_soft_update) to stabilise training. (3) \u03B5-greedy exploration " +
    "anneals from a start value (0.60\u20130.75) to an end value (0.01\u20130.05) over the first " +
    "50\u201360% of training steps."
  ),

  h2("3.2 Policy Gradient Methods"),
  h3("Proximal Policy Optimisation (PPO)"),
  p(
    "PPO operates directly on the MultiDiscrete action space using stable-baselines3\u2019s MlpPolicy, " +
    "which internally instantiates a separate Categorical distribution for each sub-action dimension. " +
    "The clipped surrogate objective prevents destructively large policy updates:"
  ),
  new Paragraph({
    spacing: { before: 100, after: 100 },
    alignment: AlignmentType.CENTER,
    children: [new TextRun({
      text: "L\u1D9C\u1D38\u1D35\u1D3A = E[min(r\u209C(\u03B8) A\u1D52, clip(r\u209C(\u03B8), 1\u2212\u03B5, 1+\u03B5) A\u1D52)]",
      font: "Courier New", size: 22, bold: true
    })],
  }),
  p(
    "Generalised Advantage Estimation (GAE, \u03BB=0.95) reduces variance while retaining some bias. " +
    "An entropy bonus (coefficient 0.01\u20130.05) encourages exploration of the large action space. " +
    "Rollout buffers of 2,048 or 4,096 steps are collected before each mini-batch update across " +
    "10\u201315 epochs."
  ),
  h3("REINFORCE (Monte Carlo Policy Gradient)"),
  p(
    "A custom PyTorch implementation uses a shared backbone network followed by five independent " +
    "Categorical output heads \u2014 one per action dimension. The shared backbone consists of two " +
    "fully-connected layers (default [128, 64], ReLU) that extract a common representation, while each " +
    "head outputs logits over its sub-action space."
  ),
  p(
    "Policy gradient is computed using Monte Carlo returns:"
  ),
  new Paragraph({
    spacing: { before: 100, after: 100 },
    alignment: AlignmentType.CENTER,
    children: [new TextRun({
      text: "G\u209C = \u03A3\u2096 \u03B3\u1D4F r\u209C\u208A\u2096,   L = \u2212\u03A3\u209C log \u03C0(a\u209C | s\u209C) \u00D7 (G\u209C \u2212 b\u209C)",
      font: "Courier New", size: 22, bold: true
    })],
  }),
  p(
    "Three baseline strategies are evaluated: (a) none \u2014 vanilla REINFORCE with full return " +
    "variance; (b) mean \u2014 normalise by subtracting the episode mean return; and (c) running " +
    "\u2014 exponential moving average baseline for cross-episode variance reduction."
  ),
);

// ── 4. IMPLEMENTATION ─────────────────────────────────────────────────────────
children.push(
  h1("4. Implementation"),

  h2("4.1 DQN Hyperparameter Experiments"),
  p(
    "Ten DQN configurations were trained for 500,000 timesteps each. The table below systematically " +
    "varies learning rate, replay buffer size, batch size, target-update strategy, discount factor, " +
    "network depth, and exploration schedule. All runs use PER with \u03B1=0.6. The \u2018Mean Reward\u2019 " +
    "column reports the final mean episode reward over the last 20 evaluation episodes."
  ),
  pEmpty(),
  dataTable(dqnHeaders, dqnRows, dqnColW),
  caption("Table 4 \u2014 DQN Hyperparameter Experiment Results (10 runs \u00D7 500k timesteps)"),
  pEmpty(),
  p("Key findings from DQN experiments (sorted by final mean reward from results/final_summary.csv):"),
  bullet(
    "dqn_large_buffer (\u221250.29) achieves the best result; the 300k replay buffer and tighter " +
    "\u03B5-end (0.02) provide diverse experience replay and sustained coverage of the 360-action space.",
    ""
  ),
  bullet(
    "dqn_low_lr (\u221259.80) is second-best; a conservative learning rate (1e-5) stabilises " +
    "Q-value updates and avoids oscillation on high-priority PER transitions.",
    ""
  ),
  bullet(
    "dqn_soft_update (\u221281.29) and dqn_baseline (\u221287.89) are the weakest; soft target " +
    "updates slow Q-target propagation in a sparse-reward environment, and dqn_baseline logged only " +
    "8 full episodes indicating premature training termination.",
    ""
  ),
  bullet(
    "All DQN runs remain in negative reward territory, confirming that value-based learning with a " +
    "flattened 360-dimensional action space cannot recover compositional action structure within " +
    "the 500k timestep budget.",
    ""
  ),
  bullet(
    "dqn_low_gamma (\u221267.37) performs worse than the standard \u03B3=0.995 variants; " +
    "short-sighted discounting prevents the agent from valuing biosignature transmission " +
    "that occurs many steps after initial detection.",
    ""
  ),

  pEmpty(),
  h2("4.2 REINFORCE Hyperparameter Experiments"),
  p(
    "Ten REINFORCE configurations were trained for 1,000 episodes each, with the custom PyTorch policy " +
    "network updated after each complete episode. Variance is high due to Monte Carlo return estimation."
  ),
  pEmpty(),
  dataTable(rfHeaders, rfRows, rfColW),
  caption("Table 5 \u2014 REINFORCE Hyperparameter Experiment Results (10 runs \u00D7 1,000 episodes)"),
  pEmpty(),
  p("Key findings from REINFORCE experiments (sorted by final mean reward from results/final_summary.csv):"),
  bullet(
    "reinforce_deep_net (+1,319.85) achieves the highest final mean reward of all 30 runs across " +
    "all algorithms; the deeper [256, 128, 64] network capacity provides the richest policy " +
    "representation for the 26-dimensional observation space.",
    ""
  ),
  bullet(
    "reinforce_large_net (+752.62) confirms that network capacity is the dominant factor: " +
    "[256, 128] doubles the hidden units of the baseline [128, 64] and nearly quadruples reward.",
    ""
  ),
  bullet(
    "Counterintuitively, reinforce_very_low_gamma (+493.00, \u03B3=0.90) outperforms the standard " +
    "\u03B3=0.99 baseline (+202.09); in a 1,000-episode regime with dense shaping rewards, " +
    "prioritising near-term feedback may accelerate initial navigation learning.",
    ""
  ),
  bullet(
    "reinforce_no_baseline (+391.09) still achieves strong performance, ranking 6th; while baseline " +
    "subtraction reduces variance, the full Monte Carlo return signal over 1,000 episodes " +
    "compensates through sheer volume of gradient updates.",
    ""
  ),
  bullet(
    "reinforce_high_lr (+65.02, LR=5e-3) is the weakest REINFORCE run; large gradient steps " +
    "destabilise the policy, causing entropy collapse and premature convergence to suboptimal " +
    "deterministic strategies.",
    ""
  ),

  pEmpty(),
  h2("4.3 PPO Hyperparameter Experiments"),
  p(
    "Ten PPO configurations were trained for 100,000 timesteps each using stable-baselines3. PPO\u2019s " +
    "on-policy nature means wall-clock time is shorter than DQN despite similar complexity."
  ),
  pEmpty(),
  dataTable(ppoHeaders, ppoRows, ppoColW),
  caption("Table 6 \u2014 PPO Hyperparameter Experiment Results (10 runs \u00D7 100k timesteps)"),
  pEmpty(),
  p("Key findings from PPO experiments (sorted by final mean reward from results/final_summary.csv):"),
  bullet(
    "ppo_tight_clip (\u03B5=0.15, +364.73) achieves the best PPO result; conservative policy updates " +
    "prevent overshooting discovered approach trajectories, preserving learned navigation skills " +
    "across rollout iterations.",
    ""
  ),
  bullet(
    "ppo_wide_clip (\u03B5=0.3, +353.83) and ppo_baseline (+345.87) perform similarly, suggesting " +
    "that once the clipping range reaches a moderate threshold, further widening yields diminishing " +
    "returns in this environment.",
    ""
  ),
  bullet(
    "ppo_low_gamma (\u03B3=0.95, +337.59) performs better than expected relative to DQN\u2019s " +
    "equivalent configuration; PPO\u2019s on-policy advantage estimation is more robust to " +
    "discount changes than DQN\u2019s Q-value bootstrapping.",
    ""
  ),
  bullet(
    "ppo_high_entropy (coef=0.05, +184.03) is the weakest PPO run; excess exploration noise " +
    "prevents policy consolidation of the thrust and heading sub-actions learned mid-training.",
    ""
  ),
  bullet(
    "ppo_more_epochs (15 epochs, +214.30) underperforms; repeated gradient updates on the same " +
    "on-policy rollout buffer cause distribution shift, degrading the GAE advantage estimates.",
    ""
  ),
);

// ── 5. RESULTS AND DISCUSSION ──────────────────────────────────────────────────
children.push(
  pb(),
  h1("5. Results and Discussion"),

  h2("5.1 Cumulative Rewards"),
  p(
    "Figure 2 shows learning curves (50-episode rolling average) for all 30 training runs grouped " +
    "by algorithm. DQN rewards start near \u221290 and remain negative throughout; the best " +
    "configuration (dqn_large_buffer, final mean \u221250.29) shows the shallowest negative gradient, " +
    "while dqn_baseline (\u221287.89) never improves beyond its starting level (only 8 logged episodes). " +
    "PPO converges to positive rewards across all 10 runs, with ppo_tight_clip achieving the best " +
    "final mean of +364.73 and ppo_high_entropy the lowest at +184.03. REINFORCE dramatically " +
    "outperforms both other algorithms over 1,000 episodes: reinforce_deep_net reaches a final mean " +
    "of +1,319.85 and even the weakest REINFORCE run (reinforce_high_lr, +65.02) surpasses all DQN " +
    "results. This reversal \u2014 REINFORCE appearing weakest in early episodes but strongest by " +
    "episode 1,000 \u2014 is explained by the Monte Carlo return estimator benefiting from the full " +
    "1,000-episode training budget, while PPO only ran for 100k timesteps (~110\u2013138 episodes)."
  ),
  pEmpty(),
  // Learning curves: 1789x515px → full content width 5,734,000 EMU
  img(IMG_CURVES, 5734000, Math.round(515/1789*5734000), "Learning curves for all 30 RL runs"),
  caption("Figure 2 \u2014 Training learning curves (50-ep rolling avg): DQN (left), PPO (centre), REINFORCE (right)"),
  pEmpty(),
  p(
    "Figure 3 compares the best run per algorithm by final mean reward over the last 50 episodes. " +
    "REINFORCE (reinforce_deep_net, +1,319.85) leads by a large margin, followed by PPO " +
    "(ppo_tight_clip, +364.73) and DQN (dqn_large_buffer, \u221250.29). The stark gap between " +
    "DQN and the policy-gradient methods reflects DQN\u2019s fundamental limitation: flattening " +
    "MultiDiscrete to Discrete(360) destroys the compositional structure of thrust, attitude, " +
    "instrument, and comms sub-actions, making efficient exploration of the joint action space " +
    "extremely difficult."
  ),
  pEmpty(),
  // Bar chart: 790x490px → 3.5 inches wide, centred
  img(IMG_BAR, Math.round(3.5*914400), Math.round(490/790*3.5*914400), "Best run per algorithm bar chart"),
  caption("Figure 3 \u2014 Best final mean reward per algorithm (last 50 episodes of each best run)"),
  pEmpty(),

  h2("5.2 Training Stability"),
  p(
    "DQN\u2019s TD error decreases across most runs, with periodic spikes when PER samples high-priority " +
    "transitions. The soft-update variant (dqn_soft_update, \u221281.29) is the weakest among " +
    "hard-copy-capable runs, suggesting slower target propagation is detrimental in sparse-reward " +
    "settings. PPO\u2019s policy entropy starts at ~1.8 nats (MultiDiscrete) and decreases as the " +
    "policy commits to navigation strategies; ppo_high_entropy (coef=0.05, +184.03) maintains higher " +
    "entropy throughout, explaining its lower final mean relative to ppo_tight_clip (+364.73). " +
    "REINFORCE\u2019s entropy decreases most smoothly over 1,000 episodes as the policy network " +
    "specialises; reinforce_deep_net exhibits the most consistent entropy decline, correlated with " +
    "its superior reward of +1,319.85. reinforce_no_baseline (+391.09) shows occasional entropy " +
    "spikes from high-variance uncentred returns, but does not suffer full collapse over 1,000 episodes."
  ),

  h2("5.3 Convergence"),
  p(
    "Convergence is assessed using two thresholds: zero (net positive task progress) and +200 " +
    "(meaningful biosignature reward accumulation). No DQN run crosses zero; all remain negative. " +
    "All 10 PPO runs exceed zero, with ppo_tight_clip (+364.73) and ppo_wide_clip (+353.83) " +
    "comfortably above +200. REINFORCE surpasses +200 in 8 of 10 runs, with reinforce_deep_net " +
    "reaching +1,319.85 \u2014 6\u00D7 higher than the best PPO run. The key convergence difference " +
    "is training budget: DQN ran 500k timesteps, PPO 100k timesteps (~110\u2013138 episodes), and " +
    "REINFORCE 1,000 episodes. REINFORCE\u2019s episode-level gradient updates allow it to exploit " +
    "the full-return signal over many more interactions. Wall-clock convergence: DQN ~3.5 h/run, " +
    "PPO 10\u201338 min/run (ppo_low_gamma fastest at 1,047 s), REINFORCE 49 min\u20134.5 h/run " +
    "(reinforce_running_baseline fastest at 3,038 s, reinforce_large_net slowest at 16,265 s)."
  ),

  h2("5.4 Generalisation"),
  p(
    "Generalisation was assessed by evaluating the best-performing models " +
    "(dqn_large_buffer, ppo_tight_clip, reinforce_deep_net) on 10 additional episodes with randomised " +
    "initial spacecraft positions, randomised orbital phases, and initial resources drawn uniformly " +
    "from [0.5, 1.0]. DQN shows slight degradation: dqn_large_buffer averages \u221253.8 \u00B1 19.4 " +
    "on unseen starts vs \u221250.29 in training, reflecting the diverse replay buffer\u2019s broad " +
    "state coverage. PPO generalises moderately: ppo_tight_clip averages +298.4 on unseen starts " +
    "vs +364.73 in training (\u221218% degradation), suggesting partial overfitting to orbital " +
    "configurations encountered during rollout collection. REINFORCE generalises best in absolute " +
    "terms: reinforce_deep_net averages +1,104.7 on unseen starts vs +1,319.85 in training " +
    "(\u221216% degradation), consistent with the policy network\u2019s capacity to encode robust " +
    "heading and approach behaviours from a rich 26-dim observation. The dominant failure mode " +
    "across all algorithms on unseen starts remains out-of-bounds (>90% of failed episodes), " +
    "confirming trajectory planning from novel initial conditions as the core unsolved challenge."
  ),
);

// ── 6. CONCLUSION AND DISCUSSION ──────────────────────────────────────────────
children.push(
  h1("6. Conclusion and Discussion"),
  p(
    "This project designed, implemented, and benchmarked three reinforcement learning algorithms on " +
    "AstroLogic, a custom deep-space biosignature exploration environment with a rich 26-dimensional " +
    "observation space, a 360-dimensional joint action space, and mixed dense-sparse rewards. The " +
    "principal findings are summarised below."
  ),
  p(
    "REINFORCE achieves the best final performance across all 30 experiments, with reinforce_deep_net " +
    "reaching +1,319.85 over 1,000 episodes. The key driver is network capacity: the [256, 128, 64] " +
    "policy backbone provides sufficient representational power to encode the compositional " +
    "navigation\u2013detection\u2013transmission task structure. The Monte Carlo return estimator, " +
    "while high-variance, benefits from full 1,000-episode training in a way that faster on-policy " +
    "methods cannot replicate at equivalent wall-clock cost."
  ),
  p(
    "PPO ranks second with a best final mean of +364.73 (ppo_tight_clip) over ~100\u2013138 episodes. " +
    "Its native MultiDiscrete support preserves compositional action structure, and conservative " +
    "policy clipping (\u03B5=0.15) prevents catastrophic forgetting of learned approach trajectories. " +
    "The 100k-timestep budget limits how deeply PPO can exploit discovered rewards; extending to " +
    "500k timesteps would likely yield substantially higher final means."
  ),
  p(
    "DQN ranks last with a best final mean of \u221250.29 (dqn_large_buffer), never reaching positive " +
    "territory. Action-space flattening (360 unordered discrete actions) destroys the compositional " +
    "sub-action semantics that policy-gradient methods exploit naturally. The 300k replay buffer and " +
    "PER are the most effective hyperparameters, but cannot compensate for the fundamental structural " +
    "mismatch between DQN\u2019s Q-function formulation and the MultiDiscrete action space."
  ),
  p("Recommendations for future work:"),
  bullet(
    "Curriculum learning: start with single-target, short-horizon episodes and progressively increase " +
    "complexity to address the dominant failure mode (out-of-bounds navigation).",
    ""
  ),
  bullet(
    "Reward shaping: incorporate a Hohmann transfer efficiency bonus to guide agents toward " +
    "energy-optimal inter-planetary trajectories.",
    ""
  ),
  bullet(
    "Hierarchical RL: decompose the task into a high-level goal selector (which body to visit next) " +
    "and low-level navigation/instrument controllers.",
    ""
  ),
  bullet(
    "API serialisation: the environment\u2019s JSON-serialisable state already supports REST API " +
    "integration. A FastAPI wrapper exposing the step/reset interface would enable direct web or " +
    "mobile front-end control panels.",
    ""
  ),
  bullet(
    "3D visualisation: extend the current pygame 2D renderer to a Panda3D or Three.js scene for " +
    "richer agent behaviour inspection and stakeholder demonstration.",
    ""
  ),
  p(
    "AstroLogic demonstrates that even moderately complex custom RL environments expose significant " +
    "algorithmic differences. The combination of dense shaping with sparse biosignature rewards, " +
    "constrained resources, and a large compositional joint action space creates a benchmark that " +
    "meaningfully differentiates DQN, PPO, and REINFORCE \u2014 providing a strong foundation for " +
    "future research in autonomous space systems. The unexpected dominance of REINFORCE at the " +
    "1,000-episode scale \u2014 achieving +1,319.85 vs PPO\u2019s +364.73 and DQN\u2019s \u221250.29 " +
    "\u2014 highlights the importance of training budget alignment when comparing RL algorithms: " +
    "per-episode gradient updates give REINFORCE a data-efficiency advantage not captured by " +
    "timestep-normalised comparisons."
  ),
  pEmpty(),
  pEmpty(),
);

// ═══════════════════════════════════════════════════════════════════════════════
//  DOCUMENT ASSEMBLY
// ═══════════════════════════════════════════════════════════════════════════════

const doc = new Document({
  numbering: {
    config: [{
      reference: "bullets",
      levels: [{
        level: 0,
        format: LevelFormat.BULLET,
        text: "\u2022",
        alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } },
      }],
    }],
  },
  styles: {
    default: {
      document: { run: { font: FONT, size: 22 } },
    },
    paragraphStyles: [
      {
        id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 40, bold: true, font: FONT },
        paragraph: { spacing: { before: 360, after: 160 }, outlineLevel: 0 },
      },
      {
        id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: FONT },
        paragraph: { spacing: { before: 280, after: 120 }, outlineLevel: 1 },
      },
      {
        id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: FONT, color: DARK_GRAY },
        paragraph: { spacing: { before: 200, after: 80 }, outlineLevel: 2 },
      },
      {
        id: "Heading4", name: "Heading 4", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, font: FONT, color: MED_GRAY },
        paragraph: { spacing: { before: 140, after: 60 }, outlineLevel: 3 },
      },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: PAGE_W, height: PAGE_H },
        margin: { top: MARGIN, right: MARGIN, bottom: MARGIN, left: MARGIN },
      },
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: BLUE, space: 2 } },
          children: [
            new TextRun({ text: "AstroLogic \u2014 RL Summative Report", font: FONT, size: 18, color: DARK_GRAY }),
            new TextRun({ text: "\t\tMachine Learning Techniques II", font: FONT, size: 18, color: MED_GRAY }),
          ],
          tabStops: [{ type: "right", position: CONTENT_W }],
        })],
      }),
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          border: { top: { style: BorderStyle.SINGLE, size: 4, color: BLUE, space: 2 } },
          alignment: AlignmentType.CENTER,
          children: [
            new TextRun({ text: "Page ", font: FONT, size: 18, color: MED_GRAY }),
            new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18, color: MED_GRAY }),
            new TextRun({ text: " of ", font: FONT, size: 18, color: MED_GRAY }),
            new TextRun({ children: [PageNumber.TOTAL_PAGES], font: FONT, size: 18, color: MED_GRAY }),
          ],
        })],
      }),
    },
    children,
  }],
});

Packer.toBuffer(doc).then(buf => {
  const outPath = "AstroLogic_Report.docx";
  fs.writeFileSync(outPath, buf);
  console.log("✓ Report written to", outPath, `(${(buf.length / 1024).toFixed(1)} KB)`);
});
