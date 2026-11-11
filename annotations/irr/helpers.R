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
  # Labels × labels for one task ("do", "see") or do vs. see ("cross"), ordered as in order_confusion
  if (source == "cross") {
    a <- aligned_counts(mains, field)
    m <- cross_task(a$do, a$see)
  } else {
    m <- coincidence(count_matrix(mains[[source]], field))
  }
  order_confusion(m)
}

order_confusion <- function(m) {
  # Labels ordered by how often each was chosen, the same on both axes; labels only one axis has go last.
  # Empty rows and columns are dropped
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

# ---- Models vs. coders ----
# Each model scored every label for every main clip (inference/vlm_label_scoring.py). Its prediction is the
# argmax label; prob is its softmax over the label set.

# Categorical slots 4, 5 and 7 of the dataviz reference palette, so do and see keep blue and orange
MODEL_COLORS <- c("qwen3.8-27b" = "#eda100", "internvl3.5-30b-a3b" = "#e87ba4", "kimi-vl-a3b" = "#6250d6")
CODERS <- "coders (each other)"
BASELINE <- "most frequent label (no video)"
RATER_COLORS <- c(setNames("#3b3b38", CODERS), setNames("#9a9992", BASELINE), MODEL_COLORS)

align_labels <- function(x, labels) {
  out <- matrix(0, nrow(x), length(labels), dimnames = list(rownames(x), labels))
  out[, colnames(x)] <- x
  out
}

one_hot <- function(p) {
  # 1 at each row's largest entry (the first one on ties), 0 elsewhere
  top <- p * 0
  top[cbind(seq_len(nrow(p)), max.col(p, "first"))] <- 1
  top
}

model_inputs <- function(main, models, task, coarse) {
  # Coder count matrix (clips with ≥2 coders that every model scored) and each model's clips × labels
  # probabilities, on one label set. coarse() merges labels as for the coders, summing their probabilities
  m <- models |> filter(task == !!task) |> mutate(label = coarse(label)) |>
    summarise(prob = sum(prob), .by = c(model, videoFilename, label))
  n <- count_matrix(main, "activity")
  scored <- reduce(map(split(m$videoFilename, m$model), unique), intersect)
  n <- n[rownames(n) %in% scored, , drop = FALSE]
  labels <- radix_sort(c(colnames(n), m$label))
  probs <- map(split(m, m$model), \(d) {
    p <- matrix(0, nrow(n), length(labels), dimnames = list(rownames(n), labels))
    d <- filter(d, videoFilename %in% rownames(n))
    p[cbind(d$videoFilename, d$label)] <- d$prob
    p
  })
  list(n = align_labels(n, labels), probs = probs[intersect(names(MODEL_COLORS), names(probs))])
}

rater_stats <- function(n, top = NULL, p = NULL) {
  # Agreement and κ of a rater with the coders on each clip; top = NULL means coders vs. each other.
  # The rest are for a rater with predictions: match with the coders' plurality (clips with one) and
  # consensus (≥80% of coders) label, and mean surprisal of the coders' labels under p
  share <- n / rowSums(n)
  if (is.null(top)) return(c(agreement = pairwise_agreement(n), kappa = matrix_kappa(coincidence(n))))
  hit <- rowSums(top * share)
  plural <- rowSums(n == apply(n, 1, max)) == 1
  cons <- apply(share, 1, max) >= 0.8
  c(agreement = mean(hit), kappa = matrix_kappa(cross_task(top, n)),
    plurality_match = mean(hit[plural] == apply(share[plural, , drop = FALSE], 1, max)),
    consensus_match = mean(hit[cons] >= 0.8),
    surprisal_bits = if (is.null(p)) NA else sum(n * -log2(p)) / sum(n))
}

baseline_rater <- function(n) {
  # Always the coders' most common label, with the coders' label frequencies as its probabilities
  freq <- colSums(n) / sum(n)
  p <- matrix(freq, nrow(n), ncol(n), byrow = TRUE, dimnames = dimnames(n))
  list(top = one_hot(p), p = p)
}

model_label_table <- function(n, top) {
  # Per label: share of coder answers and of model predictions, and recall / precision of the model
  # against the coders, each clip weighted 1
  share <- n / rowSums(n)
  hit <- colSums(share * top)
  tibble(label = colnames(n), coder_share = colSums(share) / nrow(n), model_share = colSums(top) / nrow(n),
         recall = hit / colSums(share), precision = hit / colSums(top)) |>
    filter(coder_share > 0 | model_share > 0)
}

model_pairs <- function(n, probs) {
  # Every pair of models: how often their predictions match, κ between them, and on clips with a single
  # plurality coder label, how often both chose it and how often both chose the same other label
  share <- n / rowSums(n)
  plural <- rowSums(n == apply(n, 1, max)) == 1
  plural_label <- max.col(n, "first")
  tops <- map(probs, one_hot)
  combn(names(tops), 2, simplify = FALSE) |>
    map_dfr(\(p) {
      a <- tops[[p[1]]]; b <- tops[[p[2]]]
      ia <- max.col(a, "first"); ib <- max.col(b, "first")
      tibble(model_a = p[1], model_b = p[2], agreement = mean(ia == ib), kappa = matrix_kappa(crossprod(a, b)),
             both_match_plurality = mean((ia == plural_label & ib == plural_label)[plural]),
             same_wrong_label = mean((ia == ib & ia != plural_label)[plural]))
    })
}

hard_clips <- function(n, probs) {
  # Per clip: how much the coders agree with each other and how much the models agree with the coders.
  # difficulty is 1 minus the mean of the two, so a clip that splits coders and that models miss scores highest
  share <- n / rowSums(n)
  hits <- map(probs, \(p) rowSums(one_hot(p) * share))
  preds <- map(probs, \(p) {
    i <- max.col(p, "first")
    sprintf("%s (%.2f)", colnames(p)[i], p[cbind(seq_len(nrow(p)), i)])
  })
  tibble(videoFilename = rownames(n), coders = rowSums(n),
         coder_labels = apply(n, 1, \(r) fmt_counts(sort(r, decreasing = TRUE))),
         coder_agreement = clip_agreement(n), model_agreement = rowMeans(do.call(cbind, hits)), !!!preds) |>
    mutate(difficulty = 1 - (coder_agreement + model_agreement) / 2, .after = model_agreement) |>
    arrange(desc(difficulty), coder_agreement) |>
    mutate(difficulty_rank = row_number(), .before = 1)
}

confusable_pairs <- function(n) {
  # Every pair of labels coders split between. weighted_pairs is the pair's off-diagonal cell of the coincidence
  # matrix (disagreeing coder pairs, each annotation adding 1 in total). a_to_b: when a coder says a, the share of
  # the other coders on the clip who say b. confusability is the mean of a_to_b and b_to_a
  o <- coincidence(n)
  p <- o / rowSums(o)
  ij <- which(upper.tri(o) & o > 0, arr.ind = TRUE)
  a <- rownames(o)[ij[, 1]]; b <- colnames(o)[ij[, 2]]
  tibble(label_a = a, label_b = b, weighted_pairs = o[ij], a_to_b = p[ij], b_to_a = p[ij[, 2:1, drop = FALSE]],
         clips = map2_int(a, b, \(x, y) sum(n[, x] > 0 & n[, y] > 0))) |>
    mutate(confusability = (a_to_b + b_to_a) / 2, .after = b_to_a) |>
    arrange(desc(weighted_pairs))
}

confusable_clips <- function(n, pairs) {
  # The clips behind each confusable pair: clips where both labels were chosen, most evenly split first
  map2_dfr(pairs$label_a, pairs$label_b, \(a, b) {
    both <- n[, a] > 0 & n[, b] > 0
    if (!any(both)) return(NULL)
    m <- n[both, , drop = FALSE]
    tibble(label_a = a, label_b = b, videoFilename = rownames(m), count_a = unname(m[, a]), count_b = unname(m[, b]),
           coder_labels = unname(apply(m, 1, \(r) fmt_counts(sort(r, decreasing = TRUE)))))
  }) |>
    arrange(label_a, label_b, desc(pmin(count_a, count_b)), desc(count_a + count_b))
}

short_clip <- function(f) {
  # "S00320002 2024-10-20 c0060" from a clip file name, for chart labels
  sub("^(S[0-9]+)_([0-9-]+)_.*chunk0*([0-9]+)of[0-9]+\\.mp4$", "\\1 \\2 c\\3", f)
}

interacting_review <- function(mains) {
  # "Anyone interacting" is the same question in both tasks, so each clip's do and see coders are pooled
  # (5 + 5 = 10 annotations, or 9 where see has 4 coders). Clips are ranked by pooled pairwise agreement,
  # lowest first, then by the gap between the two tasks' share answering yes
  a <- aligned_counts(mains, "anyoneInteracting")
  d <- a$do; s <- a$see
  tibble(videoFilename = rownames(d), annotations = rowSums(d) + rowSums(s), yes = d[, "yes"] + s[, "yes"],
         do_yes = d[, "yes"], do_coders = rowSums(d), see_yes = s[, "yes"], see_coders = rowSums(s)) |>
    mutate(no = annotations - yes, .after = yes) |>
    mutate(share_yes = yes / annotations,
           agreement = (yes * (yes - 1) + no * (no - 1)) / (annotations * (annotations - 1)),
           do_share_yes = do_yes / do_coders, see_share_yes = see_yes / see_coders,
           task_gap = abs(do_share_yes - see_share_yes)) |>
    arrange(agreement, desc(task_gap)) |>
    mutate(review_rank = row_number(), .before = 1)
}

# ---- Models vs. consensus labels ----

consensus_stats <- function(n, top = NULL) {
  # On clips where ≥80% of coders chose one label: accuracy against that label and κ between the rater and it.
  # top = NULL means a single coder (each coder on the clip, clip weighted 1)
  share <- n / rowSums(n)
  cons <- apply(share, 1, max) >= 0.8
  truth <- one_hot(n[cons, , drop = FALSE])
  rater <- if (is.null(top)) share[cons, , drop = FALSE] else top[cons, , drop = FALSE]
  c(clips = sum(cons), accuracy = mean(rowSums(truth * rater)), kappa = matrix_kappa(crossprod(truth, rater)))
}

consensus_by_label <- function(n, probs) {
  # Accuracy on consensus clips by consensus label, for a single coder and each model
  share <- n / rowSums(n)
  cons <- apply(share, 1, max) >= 0.8
  truth <- colnames(n)[max.col(n[cons, , drop = FALSE], "first")]
  raters <- c(list("a single coder" = share), map(probs, one_hot))
  imap_dfr(raters, \(r, name) tibble(rater = name, consensus_label = truth, hit = rowSums(one_hot(n[cons, , drop = FALSE]) * r[cons, , drop = FALSE]))) |>
    summarise(clips = n(), accuracy = mean(hit), .by = c(consensus_label, rater))
}

# ---- Human vs. model confusion matrices ----

row_props <- function(m) m / ifelse(rowSums(m) > 0, rowSums(m), 1)

offdiag_r <- function(a, b, rows) {
  # Pearson r between the off-diagonal cells of two row-proportion matrices, over the given rows
  a <- row_props(a)[rows, , drop = FALSE]; b <- row_props(b)[rows, , drop = FALSE]
  off <- outer(rows, colnames(a), `!=`)
  cor(a[off], b[off])
}

confusion_similarity <- function(n, probs, min_answers = 10, n_split = 200, seed = 0) {
  # How alike two confusion matrices are: r between their off-diagonal row proportions, i.e. where answers go
  # when they leave the coders' label. The coders' matrix is the coincidence matrix; a model's is coder label ×
  # prediction. Every comparison splits the clips into two random halves and builds one matrix from each half,
  # so no clip is in both and every r uses the same amount of data; over n_split splits it gives the mean r and
  # the middle 95%. Rows are coder labels with ≥ min_answers coder answers
  set.seed(seed)
  rows <- colnames(n)[colSums(n) >= min_answers]
  mats <- function(i) c(list(coders = coincidence(n[i, , drop = FALSE])),
                        map(probs, \(p) crossprod(n[i, , drop = FALSE] / rowSums(n[i, , drop = FALSE]), one_hot(p[i, , drop = FALSE]))))
  raters <- c("coders", names(probs))
  pairs <- c(map(raters, \(r) c(r, r)), combn(raters, 2, simplify = FALSE))
  splits <- replicate(n_split, {
    a <- sample(nrow(n), nrow(n) %/% 2)
    A <- mats(a); B <- mats(setdiff(seq_len(nrow(n)), a))
    # Different raters: average both directions (a's matrix from half A vs. b's from half B, and the reverse)
    map_dbl(pairs, \(p) if (p[1] == p[2]) offdiag_r(A[[p[1]]], B[[p[1]]], rows) else
      mean(c(offdiag_r(A[[p[1]]], B[[p[2]]], rows), offdiag_r(B[[p[1]]], A[[p[2]]], rows))))
  })
  tibble(a = map_chr(pairs, 1), b = map_chr(pairs, 2), r = rowMeans(splits, na.rm = TRUE),
         r_lo = apply(splits, 1, quantile, 0.025, na.rm = TRUE), r_hi = apply(splits, 1, quantile, 0.975, na.rm = TRUE)) |>
    mutate(comparison = case_when(a == b & a == "coders" ~ "coders vs. coders (reference)", a == b ~ "model, two halves of clips",
                                  a == "coders" ~ "coders vs. model", TRUE ~ "model vs. model"),
           pair = if_else(a == b, paste(a, "(two halves)"), paste(a, "vs.", b)), rows = length(rows), .before = 1) |>
    arrange(factor(comparison, c("coders vs. coders (reference)", "model, two halves of clips", "coders vs. model", "model vs. model")))
}

confusion_compare <- function(n, probs) {
  # Row proportions of the coders' matrix and each model's, in the coders' label order, and their difference
  h <- coincidence(n)
  lab <- rownames(order_confusion(h))
  hp <- row_props(h)[lab, lab]
  map_dfr(names(probs), \(m) {
    mp <- row_props(crossprod(n / rowSums(n), one_hot(probs[[m]])))[lab, lab]
    expand_grid(i = seq_along(lab), j = seq_along(lab)) |>
      mutate(model = m, row_label = lab[i], col_label = lab[j], row_order = i, col_order = j,
             coders = hp[cbind(i, j)], model_share = mp[cbind(i, j)], difference = model_share - coders) |>
      select(-i, -j)
  })
}

plot_difference <- function(d, title) {
  # Heatmap of model minus coders row proportions
  lim <- max(abs(d$difference), na.rm = TRUE)
  d <- mutate(d, row = factor(row_label, rev(unique(row_label[order(row_order)]))), col = factor(col_label, unique(col_label[order(col_order)])))
  ggplot(d, aes(col, row, fill = difference)) +
    geom_tile(color = "white", linewidth = 0.4) +
    geom_text(data = filter(d, abs(difference) >= 0.05), aes(label = sprintf("%+.2f", difference)), size = 2.2, color = "#3b3b38") +
    coord_equal() +
    scale_fill_gradient2(low = "#2a78d6", mid = "#f0efec", high = "#e34948", limits = c(-lim, lim)) +
    labs(title = title, x = "Model prediction / other coder", y = "Coders' label", fill = "Model − coders") +
    theme_minimal(base_size = 9) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1), panel.grid = element_blank(), legend.position = "right")
}

# ---- What would raise reliability ----

play_merged <- function(x) if_else(str_starts(x, "playing"), "playing", x)
SCHEMES <- list(
  "original labels" = identity,
  "playing … merged into playing" = play_merged,
  "looking around = no activity" = \(x) replace(x, x == "looking around", "no activity"),
  "both merges" = \(x) replace(play_merged(x), x == "looking around", "no activity"),
  "both merges + cleaning = other" = \(x) replace(replace(play_merged(x), x == "looking around", "no activity"), x == "cleaning", "other"),
  "6 broad categories" = \(x) case_when(
    str_starts(x, "playing") | x %in% c("drawing", "reading time", "dancing") ~ "play, books, art",
    x %in% c("walking", "crawling", "being moved around") ~ "moving",
    x %in% c("eating/drinking", "cooking") ~ "food",
    x %in% c("looking around", "no activity", "other", "conversing") ~ "nothing specific, talk, other",
    x == "screen time" ~ "screen time",
    TRUE ~ "care and chores")
)

scheme_alpha <- function(main, f) {
  # α and its bootstrap CI for primary activity after relabelling with f (applied to the raw labels)
  n <- count_matrix(mutate(main, activity = f(primaryActivity)), "activity")
  ci <- bootstrap_alpha(n)
  tibble(labels = ncol(n), krippendorff_alpha = kripp_alpha(n), alpha_lo = ci[1], alpha_hi = ci[2])
}

coder_drop_alpha <- function(main, field, ks = 0:6) {
  # α after leaving out the k coders who agree least with the others on their clips
  d <- mutate(main, label = .data[[field]])
  worst <- d |>
    mutate(m = n(), .by = videoFilename) |>
    mutate(same = n(), .by = c(videoFilename, label)) |>
    filter(m >= 2) |>
    summarise(agreement = mean((same - 1) / (m - 1)), .by = annotatorName) |>
    arrange(agreement)
  map_dfr(ks, \(k) {
    n <- count_matrix(filter(d, !annotatorName %in% head(worst$annotatorName, k)), "label")
    tibble(coders_dropped = k, dropped = paste(head(worst$annotatorName, k), collapse = ", "), clips = nrow(n),
           annotations = sum(n), krippendorff_alpha = kripp_alpha(n))
  })
}

majority_stability <- function(main, field, draws = 50, seed = 1) {
  # How often two disjoint groups of a clip's coders (2 vs. the rest) give the same plurality label.
  # Ties go to the first label alphabetically
  set.seed(seed)
  agree <- replicate(draws, {
    s <- main |>
      mutate(label = .data[[field]], h = rank(runif(n())) <= 2, .by = videoFilename) |>
      summarise(top = names(which.max(table(label))), .by = c(videoFilename, h)) |>
      pivot_wider(names_from = h, values_from = top) |>
      drop_na()
    mean(s$`TRUE` == s$`FALSE`)
  })
  tibble(clips = n_distinct(main$videoFilename), plurality_labels_agree = mean(agree),
         lo = unname(quantile(agree, 0.025)), hi = unname(quantile(agree, 0.975)))
}

# ---- Tables ----

show_table <- function(df, digits = 3, page_length = 10) {
  # Interactive table with numbers rounded
  DT::datatable(mutate(df, across(where(\(x) is.double(x) && !inherits(x, "POSIXt")), \(x) round(x, digits))),
                rownames = FALSE, filter = "top",
                options = list(pageLength = page_length, scrollX = TRUE, dom = "tip"))
}
