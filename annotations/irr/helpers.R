# Reliability statistics and plotting helpers for irr.qmd.
# A "count matrix" is a clips × labels matrix: how many coders gave each label to each clip.

library(tidyverse)

FIELDS <- c(activity = "Primary activity", anyoneInteracting = "Anyone interacting")
TASKS <- c("do", "see")

# Categorical slots 1-3 of the dataviz reference palette; do and see keep their colors throughout
TASK_COLORS <- c(do = "#2a78d6", see = "#eb6834", "do, looking around = no activity" = "#1baf7a")
FIELD_COLORS <- c("Primary activity" = "#2a78d6", "Anyone interacting" = "#eb6834")
GRID <- "#e8e7e3"
REF <- "#9a9992"

theme_irr <- function() {
  theme_minimal(base_size = 10) +
    theme(panel.grid.minor = element_blank(), panel.grid.major.x = element_blank(),
          panel.grid.major.y = element_line(color = GRID, linewidth = 0.4),
          legend.position = "bottom", strip.text = element_text(face = "bold"))
}

# Labels in codepoint order, as Python's sorted(); keeps label order independent of the locale
radix_sort <- function(x) sort(unique(x), method = "radix")

fmt_counts <- function(counts) {
  # "label ×n" for each nonzero entry of a named count vector, in the given order
  counts <- counts[counts > 0]
  paste0(names(counts), " ×", counts, collapse = ", ")
}

count_matrix <- function(df, field, min_raters = 2) {
  # Clips × labels count matrix, for clips with at least min_raters ratings
  n <- unclass(table(factor(df$videoFilename, radix_sort(df$videoFilename)), factor(df[[field]], levels = radix_sort(df[[field]]))))
  n <- matrix(n, nrow(n), dimnames = list(rownames(n), colnames(n)))
  n[rowSums(n) >= min_raters, , drop = FALSE]
}

# ---- Krippendorff's alpha (nominal) ----
# With n = total pairable ratings, D = observed agreeing pairs (coincidence-matrix trace) and
# n_c = ratings per label: α = 1 − (n − 1)(n − D) / (n² − Σ n_c²). Each term is a sum over clips,
# so α of any subset of clips comes from per-clip sums.

clip_terms <- function(n) {
  m <- rowSums(n)
  list(n = n, m = m, D = rowSums(n * (n - 1)) / (m - 1))
}

alpha_from_sums <- function(tot, D, nc2) {
  den <- tot^2 - nc2
  ifelse(den > 0, 1 - (tot - 1) * (tot - D) / den, NA_real_)
}

kripp_alpha <- function(n) {
  if (nrow(n) == 0) return(NA_real_)
  t <- clip_terms(n)
  alpha_from_sums(sum(t$m), sum(t$D), sum(colSums(n)^2))
}

bootstrap_alpha <- function(n, n_boot = 1000, seed = 0) {
  # 95% CI for α from resampling clips
  if (nrow(n) < 2) return(c(NA_real_, NA_real_))
  set.seed(seed)
  boots <- replicate(n_boot, kripp_alpha(n[sample.int(nrow(n), replace = TRUE), , drop = FALSE]))
  unname(quantile(boots, c(0.025, 0.975), na.rm = TRUE))
}

clip_agreement <- function(n) {
  # Per clip: share of ordered coder pairs that gave the same label
  m <- rowSums(n)
  rowSums(n * (n - 1)) / (m * (m - 1))
}

pairwise_agreement <- function(n) if (nrow(n)) mean(clip_agreement(n)) else NA_real_

specific_agreement <- function(n) {
  # Per label: when a coder picks it, the share of the other coders on the clip who pick it too
  m <- rowSums(n)
  tibble(label = colnames(n), times_chosen = colSums(n),
         specific_agreement = colSums(n * (n - 1)) / colSums(n * (m - 1))) |>
    arrange(specific_agreement)
}

cohen_kappa <- function(a, b) {
  lv <- union(a, b)
  # Undefined when both coders used one label for everything
  if (length(lv) < 2) return(NA_real_)
  p <- prop.table(table(factor(a, lv), factor(b, lv)))
  pe <- sum(rowSums(p) * colSums(p))
  (sum(diag(p)) - pe) / (1 - pe)
}

pair_kappas <- function(main, field, min_overlap) {
  # Raw agreement and Cohen's κ for every coder pair sharing at least min_overlap clips
  x <- main |> select(videoFilename, coder = annotatorName, label = all_of(field))
  inner_join(x, x, by = "videoFilename", suffix = c("_a", "_b"), relationship = "many-to-many") |>
    filter(coder_a < coder_b) |>
    summarise(shared_clips = n(), agreement = mean(label_a == label_b),
              kappa = cohen_kappa(label_a, label_b), .by = c(coder_a, coder_b)) |>
    filter(shared_clips >= min_overlap) |>
    arrange(coder_a, coder_b)
}

alpha_curve <- function(n, n_draws = 200, seed = 0) {
  # α on random subsets of k clips (without replacement), for k from 5 up to all clips
  set.seed(seed)
  sizes <- unique(as.integer(seq(min(5, nrow(n)), nrow(n), length.out = 15)))
  map_dfr(sizes, function(k) {
    draws <- replicate(n_draws, kripp_alpha(n[sample.int(nrow(n), k), , drop = FALSE]))
    # Tiny subsets where everyone gave one label have undefined α; skip them
    q <- quantile(draws, c(0.025, 0.5, 0.975), na.rm = TRUE, names = FALSE)
    tibble(clips = k, alpha_median = q[2], alpha_lo = q[1], alpha_hi = q[3])
  })
}

best_order <- function(n) {
  # Clips from most to least agreeable: greedily drop the clip whose removal raises α most, then
  # add them back in reverse. Returns each clip with α and pairwise agreement over the top k clips.
  t <- clip_terms(n)
  keep <- seq_len(nrow(n))
  dropped <- integer()
  tot <- sum(t$m); D <- sum(t$D); nc <- colSums(n)
  while (length(keep) > 2) {
    nc_wo <- sweep(-n[keep, , drop = FALSE], 2, nc, "+")
    tot_wo <- tot - t$m[keep]
    a <- alpha_from_sums(tot_wo, D - t$D[keep], rowSums(nc_wo^2))
    # α is undefined when the rest all share one label; never prefer that
    i <- which.max(replace(a, is.na(a), -Inf))
    j <- keep[i]
    tot <- tot - t$m[j]; D <- D - t$D[j]; nc <- nc - n[j, ]
    dropped <- c(dropped, j)
    keep <- keep[-i]
  }
  order <- c(keep, rev(dropped))
  cum_nc <- apply(n[order, , drop = FALSE], 2, cumsum)
  alpha_k <- alpha_from_sums(cumsum(t$m[order]), cumsum(t$D[order]), rowSums(matrix(cum_nc, length(order))^2))
  tibble(
    rank = seq_along(order),
    videoFilename = rownames(n)[order],
    labels = map_chr(order, \(j) fmt_counts(n[j, ])),
    alpha_through_here = replace(alpha_k, 1, NA),
    agreement_through_here = cummean(clip_agreement(n)[order])
  )
}

split_curves <- function(main, field, seed) {
  # α over the top k clips on a held-out 2-coder half of each clip, with clips ranked by
  # best_order on the other coders (the ranking half)
  set.seed(seed)
  m <- main |> mutate(held = rank(runif(n())) <= 2, .by = videoFilename)
  labels <- radix_sort(main[[field]])
  counts <- function(d) {
    x <- unclass(table(factor(d$videoFilename, radix_sort(d$videoFilename)), factor(d[[field]], levels = labels)))
    matrix(x, nrow(x), dimnames = list(rownames(x), colnames(x)))
  }
  rank_n <- counts(filter(m, !held))
  held_n <- counts(filter(m, held))[rownames(rank_n), , drop = FALSE]
  o <- best_order(rank_n)
  held_alpha <- map_dbl(o$rank, \(k) if (k >= 2) kripp_alpha(held_n[o$videoFilename[1:k], , drop = FALSE]) else NA)
  tibble(rank = o$rank, ranking_half = o$alpha_through_here, held_out_half = held_alpha)
}

alpha_crossing <- function(alpha, threshold = 0.8) {
  # Largest k with α over the top k clips ≥ threshold, or NA
  k <- which(alpha >= threshold)
  if (length(k)) max(k) else NA_integer_
}

# ---- Labels × labels matrices ----

coincidence <- function(n) {
  # Every ordered pair of coders on a clip, weighted 1/(m − 1) so each annotation adds 1 in total
  w <- n / (rowSums(n) - 1)
  crossprod(w, n) - diag(colSums(w), ncol(n))
}

cross_task <- function(d, s) {
  # Do labels × see labels: every do coder paired with every see coder on the clip, each clip weighted 1
  crossprod(d / rowSums(d), s / rowSums(s))
}

matrix_kappa <- function(m) {
  # Agreement beyond chance for a labels × labels matrix, with chance from its margins
  p <- m / sum(m)
  pe <- sum(rowSums(p) * colSums(p))
  (sum(diag(p)) - pe) / (1 - pe)
}

aligned_counts <- function(mains, field) {
  # Do and see count matrices on the clips both tasks rated and the union of their labels
  d <- count_matrix(mains$do, field)
  s <- count_matrix(mains$see, field)
  clips <- intersect(rownames(d), rownames(s))
  labels <- radix_sort(c(colnames(d), colnames(s)))
  align <- function(x) {
    out <- matrix(0, length(clips), length(labels), dimnames = list(clips, labels))
    out[, colnames(x)] <- x[clips, , drop = FALSE]
    out
  }
  list(do = align(d), see = align(s))
}

confusion_matrix <- function(mains, source, field) {
  # Labels × labels for one task ("do", "see") or do vs. see ("cross"), ordered by how often each
  # label was chosen; labels only one axis has go last
  if (source == "cross") {
    a <- aligned_counts(mains, field)
    m <- cross_task(a$do, a$see)
  } else {
    m <- coincidence(count_matrix(mains[[source]], field))
  }
  freq <- rowSums(m) + colSums(m)
  shared <- rowSums(m) > 0 & colSums(m) > 0
  order <- c(names(sort(freq[shared], decreasing = TRUE)), names(sort(freq[!shared], decreasing = TRUE)))
  m[intersect(order, names(which(rowSums(m) > 0))), intersect(order, names(which(colSums(m) > 0))), drop = FALSE]
}

plot_confusion <- function(m, kind, title, xlab, ylab) {
  # Heatmap of counts, row proportions or PMI for a labels × labels matrix
  vals <- switch(kind,
    count = m,
    proportion = m / rowSums(m),
    pmi = ifelse(m > 0, log2(m * sum(m) / outer(rowSums(m), colSums(m))), NA)
  )
  df <- as_tibble(as.table(vals), .name_repair = ~ c("row", "col", "value")) |>
    mutate(raw = as.vector(m), row = factor(row, rev(rownames(m))), col = factor(col, colnames(m)))
  top <- max(df$value, na.rm = TRUE)
  p <- ggplot(df, aes(col, row, fill = value)) +
    geom_tile(color = "white", linewidth = 0.4) +
    coord_equal() +
    labs(title = title, x = xlab, y = ylab,
         fill = c(count = "Weighted pairs", proportion = "Share of row", pmi = "PMI (bits)")[[kind]]) +
    theme_minimal(base_size = 9) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1), panel.grid = element_blank(), legend.position = "right")
  if (kind == "pmi") {
    lim <- max(abs(df$value), na.rm = TRUE)
    p + scale_fill_gradient2(low = "#2a78d6", mid = "#f0efec", high = "#e34948", limits = c(-lim, lim), na.value = "white")
  } else {
    label <- if (kind == "proportion") sprintf("%.2f", df$value) else
      ifelse(df$value < 0.05, "<0.1", ifelse(df$value < 10, sprintf("%.1f", df$value), sprintf("%.0f", df$value)))
    p + scale_fill_gradientn(colors = c("#ffffff", "#2a78d6", "#0d3a73"), limits = c(0, NA)) +
      geom_text(data = mutate(df, label = label) |> filter(raw > 0),
                aes(label = label, color = value > 0.55 * top), size = 2.3, show.legend = FALSE) +
      scale_color_manual(values = c(`TRUE` = "white", `FALSE` = "#3b3b38"))
  }
}

# ---- Tables ----

show_table <- function(df, digits = 3, page_length = 10) {
  # Interactive table with numbers rounded
  DT::datatable(mutate(df, across(where(\(x) is.double(x) && !inherits(x, "POSIXt")), \(x) round(x, digits))),
                rownames = FALSE, filter = "top",
                options = list(pageLength = page_length, scrollX = TRUE, dom = "tip"))
}
