const fs = require("fs");
const path = require("path");

const root = path.resolve(__dirname, "..");
const failures = [];

const requiredFiles = [
  "README.md",
  "LICENSE",
  "CHANGELOG.md",
  "CONTRIBUTING.md",
  "SECURITY.md",
  ".editorconfig",
  ".gitattributes",
  ".gitignore",
  ".github/workflows/validate.yml",
  "requirements.txt",
  "app.py",
  "train_model.py",
  "src/__init__.py",
  "src/insurance_pipeline.py",
  "AutoInsurance.csv",
  "docs/DATASET.md",
  "docs/MODEL_CARD.md",
  "docs/USAGE.md",
  "examples/sample-policy.json",
  "models/.gitkeep",
  "reports/.gitkeep"
];

const expectedColumns = [
  "Customer",
  "State",
  "Customer Lifetime Value",
  "Response",
  "Coverage",
  "Education",
  "Effective To Date",
  "EmploymentStatus",
  "Gender",
  "Income",
  "Location Code",
  "Marital Status",
  "Monthly Premium Auto",
  "Months Since Last Claim",
  "Months Since Policy Inception",
  "Number of Open Complaints",
  "Number of Policies",
  "Policy Type",
  "Policy",
  "Renew Offer Type",
  "Sales Channel",
  "Total Claim Amount",
  "Vehicle Class",
  "Vehicle Size"
];

const appFeatures = [
  "State",
  "Customer Lifetime Value",
  "Response",
  "Coverage",
  "Education",
  "EmploymentStatus",
  "Gender",
  "Income",
  "Location Code",
  "Marital Status",
  "Months Since Last Claim",
  "Months Since Policy Inception",
  "Number of Open Complaints",
  "Number of Policies",
  "Policy Type",
  "Policy",
  "Renew Offer Type",
  "Sales Channel",
  "Total Claim Amount",
  "Vehicle Class",
  "Vehicle Size"
];

function fail(message) {
  failures.push(message);
}

function readText(relativePath) {
  return fs.readFileSync(path.join(root, relativePath), "utf8");
}

function listFiles(directory) {
  const entries = fs.readdirSync(directory, { withFileTypes: true });

  return entries.flatMap((entry) => {
    const fullPath = path.join(directory, entry.name);

    if ([".git", ".venv", "venv", "node_modules", "__pycache__"].includes(entry.name)) {
      return [];
    }

    if (entry.isDirectory()) {
      return listFiles(fullPath);
    }

    return [fullPath];
  });
}

function isTextFile(filePath) {
  const textBasenames = new Set([
    ".editorconfig",
    ".gitattributes",
    ".gitignore",
    "LICENSE"
  ]);

  return [
    ".csv",
    ".js",
    ".json",
    ".md",
    ".py",
    ".txt",
    ".yml"
  ].includes(path.extname(filePath)) || textBasenames.has(path.basename(filePath));
}

for (const file of requiredFiles) {
  if (!fs.existsSync(path.join(root, file))) {
    fail(`Missing required file: ${file}`);
  }
}

if (fs.existsSync(path.join(root, "requirements.txt.txt"))) {
  fail("requirements.txt.txt should be renamed to requirements.txt.");
}

if (fs.existsSync(path.join(root, "insurance_model.pkl"))) {
  fail("Tracked insurance_model.pkl should be replaced by reproducible training outputs.");
}

if (fs.existsSync(path.join(root, "AutoInsurance.csv"))) {
  const rows = readText("AutoInsurance.csv").trim().split(/\r?\n/);
  const headers = rows[0].split(",");

  if (JSON.stringify(headers) !== JSON.stringify(expectedColumns)) {
    fail("AutoInsurance.csv headers do not match the expected schema.");
  }

  if (rows.length - 1 < 9000) {
    fail("AutoInsurance.csv should contain at least 9,000 data rows.");
  }

  const targetIndex = headers.indexOf("Monthly Premium Auto");
  const targets = rows.slice(1).map((row) => Number(row.split(",")[targetIndex]));

  if (targets.some((value) => !Number.isFinite(value))) {
    fail("Monthly Premium Auto contains non-numeric values.");
  }

  if (Math.min(...targets) < 0 || Math.max(...targets) < 100) {
    fail("Monthly Premium Auto values look out of expected range.");
  }
}

try {
  const samplePolicy = JSON.parse(readText("examples/sample-policy.json"));

  for (const feature of appFeatures) {
    if (!Object.prototype.hasOwnProperty.call(samplePolicy, feature)) {
      fail(`Sample policy missing feature: ${feature}`);
    }
  }
} catch (error) {
  fail(`Sample policy JSON is invalid: ${error.message}`);
}

const appSource = fs.existsSync(path.join(root, "app.py")) ? readText("app.py") : "";
const pipelineSource = fs.existsSync(path.join(root, "src/insurance_pipeline.py"))
  ? readText("src/insurance_pipeline.py")
  : "";

if (!appSource.includes("st.set_page_config")) {
  fail("Streamlit app should configure page settings.");
}

if (!appSource.includes("train_candidate_models")) {
  fail("Streamlit app should use the reusable training pipeline.");
}

for (const feature of appFeatures) {
  if (!pipelineSource.includes(feature)) {
    fail(`Pipeline missing feature: ${feature}`);
  }
}

for (const droppedField of ["Customer", "Effective To Date"]) {
  if (!pipelineSource.includes(droppedField)) {
    fail(`Pipeline should explicitly handle dropped field: ${droppedField}`);
  }
}

for (const dependency of ["streamlit", "pandas", "scikit-learn", "joblib"]) {
  if (!readText("requirements.txt").includes(dependency)) {
    fail(`requirements.txt missing dependency: ${dependency}`);
  }
}

for (const section of ["Quick Start", "Train and Export Metrics", "Validation", "Important Disclaimer"]) {
  if (!readText("README.md").includes(`## ${section}`)) {
    fail(`README missing section: ${section}`);
  }
}

const forbiddenPatterns = [
  { label: "mojibake text", regex: /[\u00f0\u0178\u00e2\u0161\u00ef\u00b8]/ },
  { label: "local absolute path", regex: /[A-Z]:\\/ },
  { label: "private credential", regex: /(api[_-]?key|secret|password)\s*=/i }
];

for (const filePath of listFiles(root).filter(isTextFile)) {
  const relativePath = path.relative(root, filePath).replace(/\\/g, "/");
  const content = fs.readFileSync(filePath, "utf8");

  for (const { label, regex } of forbiddenPatterns) {
    if (regex.test(content)) {
      fail(`${relativePath} contains ${label}.`);
    }
  }
}

if (failures.length > 0) {
  console.error("Project validation failed:");

  for (const failure of failures) {
    console.error(`- ${failure}`);
  }

  process.exit(1);
}

console.log("Project validation passed.");
