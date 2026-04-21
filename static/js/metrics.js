/* ===================================================================
   Metrics page - fetches /api/metrics and renders accuracy, per-class
   precision/recall/F1, and the confusion matrix.
   =================================================================== */

const BIAS_LABELS = {
  confirmation_bias: "Confirmation Bias",
  anchoring_bias: "Anchoring Bias",
  availability_heuristic: "Availability Heuristic",
  framing_effect: "Framing Effect",
  neutral: "Neutral",
};

document.addEventListener("DOMContentLoaded", async () => {
  const loading = document.getElementById("metrics-loading");
  const content = document.getElementById("metrics-content");

  try {
    const res = await fetch("/api/metrics");
    if (!res.ok) throw new Error("Metrics request failed");
    const data = await res.json();

    // Accuracy
    document.getElementById("accuracy-value").textContent =
      (data.accuracy * 100).toFixed(2) + "%";

    // Per-class table
    const tbody = document.getElementById("metrics-tbody");
    tbody.innerHTML = "";
    data.classes.forEach((cls) => {
      const row = data.per_class[cls];
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${BIAS_LABELS[cls] || cls}</td>
        <td>${(row.precision * 100).toFixed(2)}%</td>
        <td>${(row.recall * 100).toFixed(2)}%</td>
        <td>${(row.f1 * 100).toFixed(2)}%</td>
        <td>${row.support}</td>
      `;
      tbody.appendChild(tr);
    });

    // Confusion matrix
    renderConfusionMatrix(data.classes, data.confusion_matrix);

    loading.classList.add("hidden");
    content.classList.remove("hidden");
  } catch (e) {
    loading.textContent = "Could not load metrics: " + e.message;
  }
});

function renderConfusionMatrix(classes, matrix) {
  const container = document.getElementById("confusion-matrix");
  const table = document.createElement("table");
  table.className = "cm-table";

  // Header row
  const thead = document.createElement("thead");
  const headerRow = document.createElement("tr");
  headerRow.appendChild(document.createElement("th")); // corner
  classes.forEach((c) => {
    const th = document.createElement("th");
    th.textContent = BIAS_LABELS[c] || c;
    headerRow.appendChild(th);
  });
  thead.appendChild(headerRow);
  table.appendChild(thead);

  // Data rows
  const tbody = document.createElement("tbody");
  classes.forEach((trueCls, i) => {
    const tr = document.createElement("tr");
    const th = document.createElement("th");
    th.textContent = BIAS_LABELS[trueCls] || trueCls;
    tr.appendChild(th);
    classes.forEach((_, j) => {
      const td = document.createElement("td");
      td.textContent = matrix[i][j];
      if (i === j) td.classList.add("cm-diag");
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);

  container.innerHTML = "";
  container.appendChild(table);
}
