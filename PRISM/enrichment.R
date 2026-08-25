#!/usr/bin/env Rscript

# KEGG-only enrichment workflow for Tutorial 5_3.
# It uses the current KEGG REST mapping with the system curl command, so no
# large enrichment-package installation is needed in the tutorial environment.
# Usage:
# Rscript enrichment.R RESPONSE_TXT SILENT_TXT PLOT_DIR

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop(
    "Usage: Rscript enrichment.R RESPONSE_TXT SILENT_TXT PLOT_DIR",
    call. = FALSE
  )
}

response_file <- normalizePath(args[[1]], mustWork = TRUE)
silent_file <- normalizePath(args[[2]], mustWork = TRUE)
plot_dir <- args[[3]]

required_packages <- c("AnnotationDbi", "org.Hs.eg.db", "ggplot2")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages) > 0) {
  stop(
    "This tutorial requires the preinstalled R packages: ",
    paste(missing_packages, collapse = ", "),
    ".",
    call. = FALSE
  )
}
if (Sys.which("curl") == "") {
  stop("The system command 'curl' is required to retrieve KEGG annotations.", call. = FALSE)
}

suppressPackageStartupMessages({
  library(AnnotationDbi)
  library(org.Hs.eg.db)
  library(ggplot2)
})

dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)

MIN_TERM_GENES <- 10
FDR_CUTOFF <- 0.05
TOP_N_TERMS <- 15

read_gene_set <- function(path) {
  genes <- trimws(readLines(path, warn = FALSE))
  unique(toupper(genes[nzchar(genes)]))
}

map_symbols_to_entrez <- function(genes) {
  mapping <- AnnotationDbi::select(
    org.Hs.eg.db,
    keys = genes,
    columns = c("SYMBOL", "ENTREZID"),
    keytype = "SYMBOL"
  )
  mapping <- unique(mapping[!is.na(mapping$ENTREZID), c("SYMBOL", "ENTREZID")])
  unique(as.character(mapping$ENTREZID))
}

download_kegg_table <- function(endpoint) {
  output_file <- tempfile(fileext = ".tsv")
  on.exit(unlink(output_file), add = TRUE)
  status <- system2(
    "curl",
    c(
      "--fail", "--silent", "--show-error", "--connect-timeout", "15", "--max-time", "120",
      endpoint
    ),
    stdout = output_file,
    stderr = FALSE
  )
  if (status != 0) {
    stop(
      "Could not retrieve the current KEGG mapping. Check the internet connection and try again.",
      call. = FALSE
    )
  }
  read.delim(
    output_file,
    header = FALSE,
    sep = "\t",
    quote = "",
    comment.char = "",
    stringsAsFactors = FALSE
  )
}

load_kegg_annotations <- function() {
  links <- download_kegg_table("https://rest.kegg.jp/link/pathway/hsa")
  names(links) <- c("gene", "pathway")
  pathways <- download_kegg_table("https://rest.kegg.jp/list/pathway/hsa")
  names(pathways) <- c("pathway", "Description")

  links$gene <- sub("^hsa:", "", links$gene)
  links$pathway <- sub("^path:", "", links$pathway)
  pathways$pathway <- sub("^path:", "", pathways$pathway)
  pathways$Description <- sub(" - Homo sapiens \\(human\\)$", "", pathways$Description)

  list(
    term_to_genes = split(links$gene, links$pathway),
    term_names = stats::setNames(pathways$Description, pathways$pathway),
    universe = unique(links$gene)
  )
}

empty_enrichment <- function() {
  data.frame(
    ID = character(), Description = character(), GeneRatio = character(),
    BgRatio = character(), pvalue = numeric(), p.adjust = numeric(),
    qvalue = numeric(), geneID = character(), Count = integer(),
    stringsAsFactors = FALSE
  )
}

run_kegg_enrichment <- function(query_genes, annotations) {
  universe <- annotations$universe
  query_genes <- intersect(unique(as.character(query_genes)), universe)
  term_to_genes <- annotations$term_to_genes
  if (length(query_genes) == 0 || length(term_to_genes) == 0) {
    return(empty_enrichment())
  }

  term_sizes <- lengths(term_to_genes)
  tested_terms <- names(term_to_genes)[term_sizes >= MIN_TERM_GENES]
  term_to_genes <- term_to_genes[tested_terms]
  term_sizes <- term_sizes[tested_terms]
  overlap_genes <- lapply(term_to_genes, intersect, y = query_genes)
  overlap_counts <- lengths(overlap_genes)
  tested_terms <- names(overlap_counts)[overlap_counts > 0]
  if (length(tested_terms) == 0) {
    return(empty_enrichment())
  }

  term_sizes <- term_sizes[tested_terms]
  overlap_counts <- overlap_counts[tested_terms]
  overlap_genes <- overlap_genes[tested_terms]
  p_values <- phyper(
    overlap_counts - 1,
    term_sizes,
    length(universe) - term_sizes,
    length(query_genes),
    lower.tail = FALSE
  )
  result <- data.frame(
    ID = tested_terms,
    Description = unname(annotations$term_names[tested_terms]),
    GeneRatio = paste0(overlap_counts, "/", length(query_genes)),
    BgRatio = paste0(term_sizes, "/", length(universe)),
    pvalue = p_values,
    p.adjust = p.adjust(p_values, method = "BH"),
    qvalue = p.adjust(p_values, method = "BH"),
    geneID = vapply(overlap_genes, paste, collapse = "/", character(1)),
    Count = as.integer(overlap_counts),
    row.names = NULL,
    stringsAsFactors = FALSE
  )
  result <- result[!is.na(result$Description) & result$p.adjust <= FDR_CUTOFF, , drop = FALSE]
  result[order(result$p.adjust, result$pvalue), , drop = FALSE]
}

prepare_kegg_comparison <- function(response_result, silent_result) {
  response_top <- response_result[seq_len(min(TOP_N_TERMS, nrow(response_result))), , drop = FALSE]
  silent_top <- silent_result[seq_len(min(TOP_N_TERMS, nrow(silent_result))), , drop = FALSE]
  term_order <- unique(c(response_top$Description, silent_top$Description))
  if (length(term_order) == 0) {
    return(NULL)
  }

  make_plot_data <- function(data, label) {
    data <- data[data$Description %in% term_order, , drop = FALSE]
    data.frame(
      source = label,
      Description = data$Description,
      Count = as.numeric(data$Count),
      neglog10_fdr = -log10(pmax(as.numeric(data$p.adjust), .Machine$double.xmin))
    )
  }
  plot_data <- rbind(
    make_plot_data(response_result, "Dopamine response"),
    make_plot_data(silent_result, "Dopamine silent")
  )
  plot_data$source <- factor(
    plot_data$source,
    levels = c("Dopamine response", "Dopamine silent")
  )
  plot_data$Description <- factor(plot_data$Description, levels = rev(term_order))
  plot_data
}

save_kegg_comparison_plot <- function(plot_data) {
  if (is.null(plot_data) || nrow(plot_data) == 0) {
    stop("No significant KEGG terms were available for plotting.", call. = FALSE)
  }

  plot <- ggplot(plot_data, aes(x = source, y = Description)) +
    geom_point(
      aes(size = Count, fill = neglog10_fdr),
      shape = 21,
      color = "white",
      stroke = 0.35,
      alpha = 0.90
    ) +
    scale_fill_gradient(low = "grey85", high = "#D73027", name = "-log10(FDR)") +
    guides(
      fill = guide_colorbar(title = "-log10(FDR)", order = 1),
      size = guide_legend(
        title = "Gene count",
        order = 2,
        override.aes = list(shape = 21, fill = "white", color = "grey40", stroke = 0.35, alpha = 1)
      )
    ) +
    theme_bw(base_size = 11) +
    labs(title = "KEGG enrichment comparison", x = "", y = "", size = "Gene count", fill = "-log10(FDR)") +
    theme(
      plot.title = element_text(hjust = 0.5, size = 12),
      axis.text.x = element_text(angle = 30, hjust = 1, size = 10),
      axis.text.y = element_text(size = 8.5),
      panel.grid.major = element_line(linewidth = 0.25, color = "grey90"),
      panel.grid.minor = element_blank()
    )

  ggsave(
    file.path(plot_dir, "KEGG_comparison.png"),
    plot,
    width = 7.5,
    height = 5.8,
    dpi = 300
  )
}

response_genes <- read_gene_set(response_file)
silent_genes <- read_gene_set(silent_file)
response_entrez <- map_symbols_to_entrez(response_genes)
silent_entrez <- map_symbols_to_entrez(silent_genes)
if (length(response_entrez) == 0 || length(silent_entrez) == 0) {
  stop("No Entrez IDs were mapped from one or both supplied gene lists.", call. = FALSE)
}

kegg_annotations <- load_kegg_annotations()
response_results <- run_kegg_enrichment(response_entrez, kegg_annotations)
silent_results <- run_kegg_enrichment(silent_entrez, kegg_annotations)
save_kegg_comparison_plot(prepare_kegg_comparison(response_results, silent_results))
