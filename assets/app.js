(function () {
  const DATA_PATHS = {
    foundation: "data/foundation_models.json",
    taxonomy: "data/pfm_taxonomy.json",
    evaluation: "data/evaluation_benchmark.json",
    papers: "data/curated_papers.json",
    resources: "data/resources.json",
    news: "data/news.json",
  };

  const state = {
    data: null,
    surveyView: "models",
    ecosystemView: "papers",
    surveyQuery: "",
    ecosystemQuery: "",
    venue: [],
    scope: [],
    dataType: [],
    modelFilters: {
      input: [],
      base: [],
      magnification: [],
      resolution: [],
      scale: [],
    },
    openModelMenu: null,
  };

  const modelMenuLabels = {
    venue: "Venue",
    scope: "Scope",
    input: "Input",
    base: "Base Method",
    magnification: "Magnification",
    resolution: "Resolution",
    scale: "Scale",
    dataType: "Data Type",
  };

  const modelFacetLabels = {
    input: "Input Modality",
    base: "Base Method",
    magnification: "Magnification",
    resolution: "Resolution",
    scale: "Model Scale",
  };

  const scopeOptions = [
    ["extractor", "Extractor"],
    ["aggregator", "Aggregator"],
    ["hybrid", "Extractor + Aggregator"],
  ];

  const modalityLabels = {
    H: "H&E (H)",
    P: "Patch (P)",
    T: "Text (T)",
    W: "WSI stains (W)",
    I: "Images (I)",
    G: "Genes (G)",
    D: "DNA (D)",
    R: "RNA (R)",
  };

  const baseMethodLabels = {
    "Sup.": "Supervised",
  };

  const dataTypeOptions = [
    ["patches", "Patches"],
    ["image_text", "Image-text pairs"],
    ["wsi_text", "WSI-text pairs"],
    ["modality_pairs", "Modality pairs"],
    ["rna", "RNA"],
    ["dna", "DNA variants"],
    ["knowledge_graph", "Knowledge graph"],
    ["diverse_stains", "Diverse stains"],
  ];

  const resourceViewConfig = {
    toolboxes: {
      sectionIndex: 0,
      label: "Toolboxes",
      itemLabel: "tools",
    },
    datasets: {
      sectionIndex: 1,
      label: "Datasets",
      itemLabel: "datasets",
    },
    benchmarks: {
      sectionIndex: 2,
      label: "Benchmarks",
      itemLabel: "benchmarks",
    },
  };

  const taskLabels = {
    classification: "Class.",
    survival_prediction: "Surv.",
    retrieval: "Retr.",
    segmentation: "Seg.",
    patch_to_patch: "P2P",
    image_to_text: "I2T",
    text_to_image: "T2I",
    report_generation: "RG",
    vqa: "VQA",
    genetic_alteration: "GA",
    molecular_prediction: "MP",
  };

  const taskGroupLabels = {
    slide_level: "Slide-level",
    patch_level: "Patch-level",
    multimodal: "Multimodal",
    biological: "Biological",
  };

  const scoreLegend = [
    ["C", "Classification or clinical task"],
    ["F", "Fine-tuning"],
    ["Z", "Zero-shot"],
    ["I", "Image-level"],
    ["F/C", "Fine-tuning / classification"],
    ["NR", "Not reported"],
  ];

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function inlineText(value) {
    return escapeHtml(value)
      .replaceAll("&lt;sub&gt;", "<sub>")
      .replaceAll("&lt;/sub&gt;", "</sub>")
      .replaceAll("&lt;sup&gt;", "<sup>")
      .replaceAll("&lt;/sup&gt;", "</sup>");
  }

  function asLines(value) {
    if (Array.isArray(value)) return value;
    if (value === undefined || value === null) return [];
    return String(value).split("\n");
  }

  function joinLines(value) {
    return asLines(value).map(inlineText).join("<br>");
  }

  function normalize(value) {
    return String(value ?? "").toLowerCase();
  }

  function includesQuery(parts, query) {
    if (!query) return true;
    return parts.some((part) => normalize(part).includes(query));
  }

  function uniqueValues(values) {
    return [...new Set(values.filter(Boolean))];
  }

  function splitModelNames(value) {
    return asLines(value).map((item) => item.trim()).filter(Boolean);
  }

  function splitTokenList(value, pattern = /[,/]+/) {
    return String(value ?? "")
      .split(pattern)
      .map((item) => item.trim())
      .filter(Boolean);
  }

  function taxonomyMatchesForModel(modelName, taxonomy) {
    const direct = taxonomy.get(modelName);
    if (direct) return [direct];
    return uniqueValues(splitModelNames(modelName).map((name) => taxonomy.get(name)));
  }

  function parseMagnificationAndResolution(value) {
    const text = String(value ?? "").trim();
    if (!text || text.toLowerCase() === "unknown") return { magnification: ["Unknown"], resolution: ["Unknown"] };

    const magnification = [];
    const resolution = [];
    let previousHadResolution = false;
    text.split(",").forEach((rawPart) => {
      const part = rawPart.trim();
      if (!part) return;
      if (part.includes("/")) {
        const [magPart, resolutionPart] = part.split("/");
        magnification.push(...splitTokenList(magPart, /[,]+/));
        resolution.push(...splitTokenList(resolutionPart, /[,]+/));
        previousHadResolution = true;
        return;
      }
      if (previousHadResolution && /^\d{3,}$/.test(part)) {
        resolution.push(part);
      } else {
        magnification.push(part);
      }
    });

    return {
      magnification: uniqueValues(magnification),
      resolution: uniqueValues(resolution),
    };
  }

  function sortFacetValues(values, facet) {
    const scaleOrder = ["XS", "S", "B", "L", "H", "g", "G", "Unknown"];
    return uniqueValues(values).sort((a, b) => {
      if (facet === "scale") {
        const aRank = scaleOrder.includes(a) ? scaleOrder.indexOf(a) : 98;
        const bRank = scaleOrder.includes(b) ? scaleOrder.indexOf(b) : 98;
        if (aRank !== bRank) return aRank - bRank;
        return a.localeCompare(b);
      }
      const aNum = Number(a);
      const bNum = Number(b);
      if (Number.isFinite(aNum) && Number.isFinite(bNum)) return aNum - bNum;
      if (a === "Unknown") return 1;
      if (b === "Unknown") return -1;
      return a.localeCompare(b);
    });
  }

  function facetTokens(taxes, facet) {
    return uniqueValues(
      taxes.flatMap((tax) => {
        if (facet === "input") return splitTokenList(tax.model_pretraining?.input, /[,]+/);
        if (facet === "base") return splitTokenList(tax.model_pretraining?.base_method, /[+/]+/);
        if (facet === "magnification") {
          return parseMagnificationAndResolution(tax.model_pretraining?.magnification_or_resolution).magnification;
        }
        if (facet === "resolution") {
          return parseMagnificationAndResolution(tax.model_pretraining?.magnification_or_resolution).resolution;
        }
        if (facet === "scale") return splitTokenList(tax.model_design?.scale, /[,/]+/);
        return [];
      })
    );
  }

  function matchesFacetFilters(taxes) {
    return Object.entries(state.modelFilters).every(([facet, selectedValues]) => {
      if (!selectedValues.length) return true;
      const tokens = facetTokens(taxes, facet);
      return selectedValues.some((value) => tokens.includes(value));
    });
  }

  function modelDataTypeTokens(model) {
    const text = normalize([...asLines(model.data_source), ...asLines(model.data_statistics)].join(" "));
    const tokens = [];
    if (/\bpatch(?:es)?\b/.test(text)) tokens.push("patches");
    if (/image[-\s]?text\s+pairs?/.test(text)) tokens.push("image_text");
    if (/wsi[-\s]?text\s+pairs?/.test(text)) tokens.push("wsi_text");
    if (/modality\s+pairs?/.test(text)) tokens.push("modality_pairs");
    if (/\brna\b/.test(text)) tokens.push("rna");
    if (/\bdna\b/.test(text)) tokens.push("dna");
    if (/\bkg\b|knowledge\s+graph/.test(text)) tokens.push("knowledge_graph");
    if (/diverse\s+stains?/.test(text)) tokens.push("diverse_stains");
    return uniqueValues(tokens);
  }

  function matchesDataTypeFilters(model) {
    if (!state.dataType.length) return true;
    const tokens = modelDataTypeTokens(model);
    return state.dataType.some((value) => tokens.includes(value));
  }

  async function loadJson(path) {
    const response = await fetch(path);
    if (!response.ok) throw new Error(`Failed to load ${path}`);
    return response.json();
  }

  async function loadData() {
    const [foundation, taxonomy, evaluation, papers, resources, news] = await Promise.all([
      loadJson(DATA_PATHS.foundation),
      loadJson(DATA_PATHS.taxonomy),
      loadJson(DATA_PATHS.evaluation),
      loadJson(DATA_PATHS.papers),
      loadJson(DATA_PATHS.resources),
      loadJson(DATA_PATHS.news),
    ]);
    return { foundation, taxonomy, evaluation, papers, resources, news };
  }

  function taxonomyByModel() {
    const map = new Map();
    state.data.taxonomy.models.forEach((item) => map.set(item.model, item));
    return map;
  }

  function modelScope(item) {
    const scope = item?.model_scope;
    if (!scope) return "Not listed";
    if (scope.extractor && scope.aggregator) return "Extractor + Aggregator";
    if (scope.extractor) return "Extractor";
    if (scope.aggregator) return "Aggregator";
    return "Not listed";
  }

  function modelScopeKey(item) {
    const scope = item?.model_scope;
    if (!scope) return "unknown";
    if (scope.extractor && scope.aggregator) return "hybrid";
    if (scope.extractor) return "extractor";
    if (scope.aggregator) return "aggregator";
    return "unknown";
  }

  function combinedModelScope(taxes) {
    if (!taxes.length) return "Not listed";
    return uniqueValues(taxes.map(modelScope)).join(" / ");
  }

  function combinedScopeKeys(taxes) {
    return uniqueValues(taxes.map(modelScopeKey));
  }

  function linksHtml(links) {
    return `<span class="link-list">${links
      .map(
        (link) => {
          const external = /^https?:\/\//i.test(link.url);
          return `<a class="link-pill" href="${escapeHtml(link.url)}" ${
            external ? 'target="_blank" rel="noreferrer"' : ""
          }>${escapeHtml(
            link.label
          )}</a>`;
        }
      )
      .join("")}</span>`;
  }

  function tagList(tags, variant = "") {
    return `<span class="tag-list">${tags
      .filter(Boolean)
      .map((tag) => `<span class="tag ${variant}">${inlineText(tag)}</span>`)
      .join("")}</span>`;
  }

  function setStats() {
    const models = state.data.foundation.models.length;
    const evalTasks = Object.values(state.data.evaluation.tasks).reduce((total, tasks) => total + tasks.length, 0);
    const papers = state.data.papers.sections.reduce((total, section) => total + section.papers.length, 0);
    const resourcesByTitle = new Map(
      state.data.resources.sections.map((section) => [
        section.title,
        section.categories.reduce((total, category) => total + category.items.length, 0),
      ])
    );

    if ($("#modelCount")) $("#modelCount").textContent = models;
    if ($("#evalTaskCount")) $("#evalTaskCount").textContent = evalTasks;
    if ($("#paperCount")) $("#paperCount").textContent = papers;
    if ($("#toolCount")) $("#toolCount").textContent = resourcesByTitle.get("🔧 Useful Toolboxes") ?? 0;
    if ($("#datasetCount")) $("#datasetCount").textContent = resourcesByTitle.get("📊 Datasets") ?? 0;
    if ($("#benchmarkCount")) $("#benchmarkCount").textContent = resourcesByTitle.get("🏆 Benchmarks") ?? 0;
  }

  function renderNews() {
    const root = $("#newsList");
    if (!root) return;
    root.innerHTML = state.data.news.items
      .map(
        (item) => `<article class="news-item">
          <time>${escapeHtml(item.date)}</time>
          <strong>${inlineText(item.title)}</strong>
          <p>${inlineText(item.description)}</p>
          ${item.links?.length ? linksHtml(item.links) : ""}
        </article>`
      )
      .join("");
  }

  function populateFilters() {
    renderModelFacetFilters();
  }

  function modelFacetOptions() {
    const models = state.data.taxonomy.models;
    return {
      input: sortFacetValues(models.flatMap((tax) => facetTokens([tax], "input")), "input"),
      base: sortFacetValues(models.flatMap((tax) => facetTokens([tax], "base")), "base"),
      magnification: sortFacetValues(models.flatMap((tax) => facetTokens([tax], "magnification")), "magnification"),
      resolution: sortFacetValues(models.flatMap((tax) => facetTokens([tax], "resolution")), "resolution"),
      scale: sortFacetValues(models.flatMap((tax) => facetTokens([tax], "scale")), "scale"),
    };
  }

  function renderModelFacetFilters() {
    const root = $("#modelFilters");
    if (!root) return;
    const menuOrder = ["venue", "scope", "input", "base", "magnification", "resolution", "scale", "dataType"];
    const hasActiveFilters =
      state.venue.length ||
      state.scope.length ||
      state.dataType.length ||
      Object.values(state.modelFilters).some((values) => values.length);
    root.innerHTML = `<div class="filter-strip">
      ${menuOrder
        .map(
          (menu) => `<button class="filter-menu-button ${
            state.openModelMenu === menu ? "active" : ""
          }" type="button" data-filter-menu="${escapeHtml(menu)}">
            <span>${escapeHtml(modelMenuLabels[menu])}</span>
            <strong>${escapeHtml(filterButtonSummary(menu))}</strong>
          </button>`
        )
        .join("")}
      <button class="filter-clear" id="clearModelFilters" type="button" ${hasActiveFilters ? "" : "disabled"}>Clear</button>
    </div>
    ${
      state.openModelMenu
        ? `<div class="filter-popover" data-open-filter="${escapeHtml(state.openModelMenu)}">
            ${filterPopoverHtml(state.openModelMenu)}
          </div>`
        : ""
    }`;
  }

  function filterButtonSummary(menu) {
    if (menu === "venue") return selectionSummary(state.venue);
    if (menu === "scope") return selectionSummary(state.scope.map((value) => labelForOption(scopeOptions, value)));
    if (menu === "input") return selectionSummary(state.modelFilters.input.map(labelModality));
    if (menu === "base") return selectionSummary(state.modelFilters.base.map(labelBaseMethod));
    if (menu in modelFacetLabels) return selectionSummary(state.modelFilters[menu]);
    if (menu === "dataType") return selectionSummary(state.dataType.map((value) => labelForOption(dataTypeOptions, value)));
    return "All";
  }

  function selectionSummary(values) {
    if (!values.length) return "All";
    if (values.length === 1) return values[0];
    return `${values.length} selected`;
  }

  function labelForOption(options, value) {
    return options.find((item) => item[0] === value)?.[1] ?? value;
  }

  function labelModality(value) {
    return modalityLabels[value] ?? value;
  }

  function labelBaseMethod(value) {
    return baseMethodLabels[value] ?? value;
  }

  function formatBaseMethodValue(value) {
    return String(value ?? "").replace(/\bSup\.(?=$|[+/,\s])/g, "Supervised").replaceAll("+", " + ");
  }

  function selectableButtonHtml(group, value, label, active) {
    return `<button class="filter-option ${active ? "active" : ""}" type="button" data-filter-group="${escapeHtml(
      group
    )}" data-filter-value="${escapeHtml(value)}">${escapeHtml(label)}</button>`;
  }

  function filterPopoverHtml(menu) {
    if (menu === "venue") {
      const venues = [...new Set(state.data.foundation.models.map((item) => item.venue))].sort();
      return optionPanelHtml("venue", venues.map((value) => [value, value]), state.venue);
    }
    if (menu === "scope") {
      return optionPanelHtml("scope", scopeOptions, state.scope);
    }
    if (menu in modelFacetLabels) {
      const options = modelFacetOptions()[menu].map((value) => [
        value,
        menu === "input" ? labelModality(value) : menu === "base" ? labelBaseMethod(value) : value,
      ]);
      return optionPanelHtml(menu, options, state.modelFilters[menu]);
    }
    if (menu === "dataType") {
      return optionPanelHtml("dataType", dataTypeOptions, state.dataType);
    }
    return "";
  }

  function optionPanelHtml(group, options, selectedValues) {
    return `<div class="filter-options">
      ${options
        .map(([value, label]) => selectableButtonHtml(group, value, label, selectedValues.includes(value)))
        .join("")}
    </div>`;
  }

  function pretrainingTags(taxes) {
    return uniqueValues(
      taxes.flatMap((tax) => {
        const magRes = parseMagnificationAndResolution(tax.model_pretraining?.magnification_or_resolution);
        return [
          tax.model_pretraining?.input && `Input: ${tax.model_pretraining.input}`,
          tax.model_pretraining?.base_method && `Base Method: ${formatBaseMethodValue(tax.model_pretraining.base_method)}`,
          tax.model_pretraining?.magnification_or_resolution && `Magnification: ${magRes.magnification.join(", ")}`,
          tax.model_pretraining?.magnification_or_resolution && `Resolution: ${magRes.resolution.join(", ")}`,
        ];
      })
    );
  }

  function renderModels() {
    const tableBody = $("#modelTable tbody");
    if (!tableBody) return;
    const taxonomy = taxonomyByModel();
    const rows = state.data.foundation.models
      .map((model) => ({ model, taxes: taxonomyMatchesForModel(model.model, taxonomy) }))
      .filter(({ model, taxes }) => {
        const haystack = [
          model.model,
          model.venue,
          ...asLines(model.method),
          ...asLines(model.architecture),
          ...asLines(model.data_source),
          ...asLines(model.data_statistics),
          ...taxes.flatMap((tax) => [
            tax.model_pretraining?.input,
            tax.model_pretraining?.base_method,
            tax.model_pretraining?.magnification_or_resolution,
            tax.model_design?.architecture,
            tax.model_design?.parameters,
            tax.model_design?.scale,
          ]),
        ];
        return (
          includesQuery(haystack, state.surveyQuery) &&
          (!state.venue.length || state.venue.includes(model.venue)) &&
          (!state.scope.length || state.scope.some((value) => combinedScopeKeys(taxes).includes(value))) &&
          matchesFacetFilters(taxes) &&
          matchesDataTypeFilters(model)
        );
      });

    tableBody.innerHTML = rows
      .map(({ model, taxes }) => {
        const pretraining = pretrainingTags(taxes);
        const architecture = [
          ...asLines(model.architecture),
          ...uniqueValues(taxes.map((tax) => tax.model_design?.parameters && `Params: ${tax.model_design.parameters}`)),
          ...uniqueValues(taxes.map((tax) => tax.model_design?.scale && `Scale: ${tax.model_design.scale}`)),
        ].filter(Boolean);
        return `<tr>
          <td><span class="model-name">${inlineText(model.model)}</span></td>
          <td>${model.venue_highlight ? `<strong>${escapeHtml(model.venue)}</strong>` : escapeHtml(model.venue)}</td>
          <td>${tagList([combinedModelScope(taxes)], "teal")}</td>
          <td>${tagList(pretraining)}</td>
          <td>${joinLines(architecture)}</td>
          <td>${joinLines(model.data_source)}<br><span class="muted-lines">${joinLines(model.data_statistics)}</span></td>
          <td>${linksHtml(model.links)}</td>
        </tr>`;
      })
      .join("");
  }

  function scoreClass(value) {
    if (value === "❌" || value === "NR") return "missing";
    if (String(value).includes("Z")) return "zero-shot";
    if (String(value).includes("F")) return "finetune";
    return "positive";
  }

  function scoreDisplay(value) {
    return value === "❌" ? "NR" : value;
  }

  function scoreTitle(value) {
    return value === "❌" || value === "NR" ? "Not reported" : value;
  }

  function renderEvaluation() {
    const evaluationTable = $("#evaluationTable");
    if (!evaluationTable) return;
    const evalLegend = $("#evalLegend");
    if (evalLegend) {
      evalLegend.innerHTML = scoreLegend
        .map(
          ([score, label]) =>
            `<span class="legend-item"><span class="score-pill ${scoreClass(score)}">${escapeHtml(
              score
            )}</span>${escapeHtml(label)}</span>`
        )
        .join("");
    }

    const groups = state.data.evaluation.tasks;
    const groupEntries = Object.entries(groups);
    const models = state.data.evaluation.models.filter((model) => includesQuery([model.model], state.surveyQuery));
    const firstHeader = `<tr><th rowspan="2">Model</th>${groupEntries
      .map(([group, tasks]) => `<th colspan="${tasks.length}">${escapeHtml(taskGroupLabels[group] ?? group)}</th>`)
      .join("")}</tr>`;
    const secondHeader = `<tr>${groupEntries
      .flatMap(([, tasks]) => tasks)
      .map((task) => `<th>${escapeHtml(taskLabels[task] ?? task)}</th>`)
      .join("")}</tr>`;
    const body = models
      .map((model) => {
        const cells = groupEntries
          .flatMap(([group, tasks]) => tasks.map((task) => model[group][task]))
          .map(
            (value) =>
              `<td><span class="score-pill ${scoreClass(value)}" title="${escapeHtml(scoreTitle(value))}">${escapeHtml(
                scoreDisplay(value)
              )}</span></td>`
          )
          .join("");
        return `<tr><td><span class="model-name">${inlineText(model.model)}</span></td>${cells}</tr>`;
      })
      .join("");

    evaluationTable.innerHTML = `<thead>${firstHeader}${secondHeader}</thead><tbody>${body}</tbody>`;
  }

  function formatDateRange(section) {
    if (!section.conference_start) return String(section.year);
    const start = new Date(`${section.conference_start}T00:00:00`);
    const end = section.conference_end ? new Date(`${section.conference_end}T00:00:00`) : null;
    const startText = start.toLocaleDateString("en", { month: "short", day: "numeric", year: "numeric" });
    if (!end) return startText;
    const endText = end.toLocaleDateString("en", { month: "short", day: "numeric", year: "numeric" });
    return `${startText} - ${endText}`;
  }

  function tabSummaryHtml(items) {
    return items
      .map(
        ([value, label]) => `<span class="summary-stat">
          <strong>${escapeHtml(value)}</strong>
          <small>${escapeHtml(label)}</small>
        </span>`
      )
      .join("");
  }

  function renderPapers() {
    const paperList = $("#paperList");
    if (!paperList) return;
    const sections = [...state.data.papers.sections].sort((a, b) =>
      String(b.conference_start ?? `${b.year}-12-31`).localeCompare(String(a.conference_start ?? `${a.year}-12-31`))
    );
    const totalPapers = sections.reduce((total, section) => total + section.papers.length, 0);
    const populatedSections = sections.filter((section) => section.papers.length).length;
    const matchingPapers = sections.reduce(
      (total, section) =>
        total +
        section.papers.filter((paper) =>
          includesQuery([paper.title, paper.summary, paper.venue, paper.year, paper.paper_url], state.ecosystemQuery)
        ).length,
      0
    );
    const paperSummary = $("#paperSummary");
    if (paperSummary) {
      paperSummary.innerHTML = tabSummaryHtml([
        [state.ecosystemQuery ? matchingPapers : totalPapers, state.ecosystemQuery ? "matching papers" : "listed papers"],
        [populatedSections, "populated sections"],
        [sections.length - populatedSections, "prepared sections"],
      ]);
    }
    paperList.innerHTML = sections
      .map((section) => {
        const papers = section.papers.filter((paper) =>
          includesQuery([paper.title, paper.summary, paper.venue, paper.year, paper.paper_url], state.ecosystemQuery)
        );
        const content = section.papers.length
          ? papers.length
            ? `<div class="paper-items">${papers
                .map(
                  (paper) => `<article class="paper-item">
                    <h4><a href="${escapeHtml(paper.paper_url)}" target="_blank" rel="noreferrer">${inlineText(
                    paper.title
                  )}</a></h4>
                    <p>${inlineText(paper.summary)}</p>
                  </article>`
                )
                .join("")}</div>`
            : `<div class="empty-state">No papers match the current search.</div>`
          : `<div class="empty-state">Coming soon.</div>`;
        return `<section class="conference-section">
          <div class="conference-head">
            <h3>${escapeHtml(section.venue)} ${escapeHtml(section.year)}</h3>
            <span class="conference-date">${escapeHtml(formatDateRange(section))}</span>
          </div>
          ${content}
        </section>`;
      })
      .join("");
  }

  function sectionLabel(title) {
    return title.replace(/^[^\w]+/u, "").trim();
  }

  function renderResourceSection(view, config) {
    const section = state.data.resources.sections[config.sectionIndex];
    const grid = $(`[data-resource-grid="${view}"]`);
    const summary = $(`[data-resource-summary="${view}"]`);
    if (!section || !grid) return;
    const totalItems = section.categories.reduce((total, category) => total + category.items.length, 0);
    let matchingItems = 0;

    const categoriesHtml = section.categories
      .map((category) => {
        const items = category.items.filter((item) =>
          includesQuery([item.name, item.description, category.title, section.title], state.ecosystemQuery)
        );
        matchingItems += items.length;
        return `<section class="resource-category">
          <div class="resource-category-head">
            <h3>${escapeHtml(category.title)}</h3>
            <span>${escapeHtml(items.length)} / ${escapeHtml(category.items.length)}</span>
          </div>
          ${
            items.length
              ? `<div class="resource-items">${items
                  .map(
                    (item) => `<a class="resource-card" href="${escapeHtml(item.url)}" target="_blank" rel="noreferrer">
                      <strong>${inlineText(item.name)}</strong>
                      <p>${inlineText(item.description)}</p>
                    </a>`
                  )
                  .join("")}</div>`
              : `<div class="empty-state">No resources match the current search.</div>`
          }
        </section>`;
      })
      .join("");

    if (summary) {
      summary.innerHTML = tabSummaryHtml([
        [state.ecosystemQuery ? matchingItems : totalItems, state.ecosystemQuery ? `matching ${config.itemLabel}` : `listed ${config.itemLabel}`],
        [section.categories.length, "groups"],
        [sectionLabel(section.title), "section"],
      ]);
    }
    grid.innerHTML = categoriesHtml;
  }

  function renderResources() {
    Object.entries(resourceViewConfig).forEach(([view, config]) => renderResourceSection(view, config));
  }

  function resourceTargetFromHash(hash) {
    const target = String(hash || "").replace("#", "").toLowerCase();
    if (target === "papers") return { view: "papers" };
    if (target in resourceViewConfig) return { view: target };
    if (target === "resources") return { view: "toolboxes" };
    return null;
  }

  function setNavActive(view) {
    $$(".top-nav a").forEach((link) => {
      const linkTarget = link.getAttribute("href").split("#").pop();
      link.classList.toggle("active", linkTarget === view);
    });
  }

  function activateResourceTarget(hash, shouldScroll = false) {
    const target = resourceTargetFromHash(hash);
    if (!target) return false;
    activateEcosystemView(target.view);
    setNavActive(target.view);
    if (shouldScroll) {
      ($("#resource-explorer") ?? $("#ecosystem"))?.scrollIntoView({ behavior: "smooth", block: "start" });
    }
    return true;
  }

  function renderAll() {
    renderNews();
    renderModels();
    renderEvaluation();
    renderPapers();
    renderResources();
  }

  function activateSurveyView(view) {
    state.surveyView = view;
    $$("[data-survey-view]").forEach((tab) => tab.classList.toggle("active", tab.dataset.surveyView === view));
    $$("[data-survey-panel]").forEach((panel) =>
      panel.classList.toggle("active", panel.dataset.surveyPanel === view)
    );
    setNavActive("survey");
  }

  function activateEcosystemView(view) {
    state.ecosystemView = view;
    $$("[data-ecosystem-view]").forEach((tab) =>
      tab.classList.toggle("active", tab.dataset.ecosystemView === view)
    );
    $$("[data-ecosystem-panel]").forEach((panel) =>
      panel.classList.toggle("active", panel.dataset.ecosystemPanel === view)
    );
  }

  function bindEvents() {
    $("#surveySearch")?.addEventListener("input", (event) => {
      state.surveyQuery = normalize(event.target.value.trim());
      renderModels();
      renderEvaluation();
    });

    $("#ecosystemSearch")?.addEventListener("input", (event) => {
      state.ecosystemQuery = normalize(event.target.value.trim());
      renderPapers();
      renderResources();
    });

    $("#modelFilters")?.addEventListener("click", (event) => {
      const clearButton = event.target.closest("#clearModelFilters");
      if (clearButton) {
        state.venue = [];
        state.scope = [];
        state.dataType = [];
        Object.keys(state.modelFilters).forEach((facet) => {
          state.modelFilters[facet] = [];
        });
        state.openModelMenu = null;
        renderModelFacetFilters();
        renderModels();
        return;
      }

      const menuButton = event.target.closest("[data-filter-menu]");
      if (menuButton) {
        const menu = menuButton.dataset.filterMenu;
        state.openModelMenu = state.openModelMenu === menu ? null : menu;
        renderModelFacetFilters();
        return;
      }

      const optionButton = event.target.closest("[data-filter-group]");
      if (optionButton) {
        const group = optionButton.dataset.filterGroup;
        const value = optionButton.dataset.filterValue;
        const selectedValues =
          group === "venue" ? state.venue : group === "scope" ? state.scope : group === "dataType" ? state.dataType : state.modelFilters[group];
        if (!selectedValues) return;
        const nextValues = selectedValues.includes(value)
          ? selectedValues.filter((item) => item !== value)
          : [...selectedValues, value];
        if (group === "venue") state.venue = nextValues;
        else if (group === "scope") state.scope = nextValues;
        else if (group === "dataType") state.dataType = nextValues;
        else state.modelFilters[group] = nextValues;
        renderModelFacetFilters();
        renderModels();
      }
    });

    $$("[data-survey-view]").forEach((tab) => {
      tab.addEventListener("click", () => activateSurveyView(tab.dataset.surveyView));
    });

    $$("[data-ecosystem-view]").forEach((tab) => {
      tab.addEventListener("click", () => {
        const view = tab.dataset.ecosystemView;
        activateEcosystemView(view);
        setNavActive(view);
        if (resourceTargetFromHash(view)) history.pushState(null, "", `#${view}`);
      });
    });

    $$(".top-nav a").forEach((link) => {
      link.addEventListener("click", (event) => {
        const view = link.getAttribute("href").replace("#", "");
        if (["models", "evaluation"].includes(view)) {
          event.preventDefault();
          activateSurveyView(view);
          $("#survey-explorer")?.scrollIntoView({ behavior: "smooth", block: "start" });
        }
        if (resourceTargetFromHash(view)) {
          event.preventDefault();
          history.pushState(null, "", `#${view}`);
          activateResourceTarget(view, false);
        }
      });
    });

    window.addEventListener("hashchange", () => {
      activateResourceTarget(window.location.hash, false);
    });

    $("#copyCitation")?.addEventListener("click", async () => {
      const citation = $("#citationText").textContent;
      try {
        await navigator.clipboard.writeText(citation);
        $("#copyCitation").textContent = "Copied";
        window.setTimeout(() => {
          $("#copyCitation").textContent = "Copy";
        }, 1400);
      } catch {
        $("#copyCitation").textContent = "Select text";
      }
    });
  }

  async function init() {
    try {
      state.data = await loadData();
      setStats();
      populateFilters();
      bindEvents();
      renderAll();
      if (!activateResourceTarget(window.location.hash, false)) {
        setNavActive($("#resource-explorer") ? "papers" : "survey");
      }
    } catch (error) {
      document.body.insertAdjacentHTML(
        "afterbegin",
        `<div class="load-error">Failed to load site data. Run a local server from the repository root and open <code>http://localhost:8000/</code>.</div>`
      );
      console.error(error);
    }
  }

  init();
})();
