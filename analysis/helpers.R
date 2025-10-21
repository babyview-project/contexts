
location_color <- "#6B8E23"
activity_color <- "#f6b26b"
activity_palette <- ggthemes::solarized_pal()(8)
top_activities <- c("cleaning", "cooking", "eating", "exploring", "looking at device", 
                    "playing", "reading", "watching tv")

summarized_data <- function(data, x_var, y_var, group_var) {
  return(data %>%
           group_by_at(c(x_var, group_var)) %>%
           summarise(mean_value = mean(.data[[y_var]], na.rm = TRUE),
                     sd_value = sd(.data[[y_var]], na.rm = TRUE),
                     n = n(),
                     se = sd_value / sqrt(n()),
                     ci_lower = mean_value - qt(1 - (0.05 / 2), n - 1) * se,
                     ci_upper = mean_value + qt(1 - (0.05 / 2), n - 1) * se,
                     .groups = 'drop')
  )
}

bin_age <- function(df) {
  df |> mutate(age_bin = case_when(
    age < 12*30 ~ "5-12",
    age < 18*30 ~ "12-18",
    age < 24*30 ~ "18-24",
    age < 30*30 ~ "24-30",
    age < 36*30 ~ "30-36")) |> group_by(age_bin, subject_id) |> filter(!is.na(age_bin)) |>
    mutate(total_count = n(), 
           age_bin = factor(age_bin, levels = c("5-12", "12-18", "18-24", "24-30", "30-36")))
}

weighted_ci_normal_df <- function(df, value_col, weight_col, group_col = NULL, conf_level = 0.95) {
  z <- qnorm(1 - (1 - conf_level) / 2)
  
  # Group if needed, otherwise treat as single group
  if (!is.null(group_col)) {
    df <- df %>% group_by(.data[[group_col]])
  }
  
  df %>%
    summarise(
      weighted_mean = weighted.mean(.data[[value_col]], .data[[weight_col]], na.rm = TRUE),
      w_var = sum(.data[[weight_col]] * (.data[[value_col]] - weighted_mean)^2, na.rm = TRUE) / sum(.data[[weight_col]], na.rm = TRUE),
      w_se = sqrt(w_var / n()),
      ci_lower = weighted_mean - z * w_se,
      ci_upper = weighted_mean + z * w_se,
      n_group = n(),
      .groups = 'drop'
    )
}

base_theme <- theme(
  # Axis text
  axis.text.x = element_text(angle = 45, hjust = 1, size = 14),  # X-axis labels
  axis.text.y = element_text(size = 14),                         # Y-axis labels
  
  # Axis titles
  axis.title.x = element_text(size = 16),
  axis.title.y = element_text(size = 16),
  
  # Legend
  legend.position = "right",
  legend.title = element_text(size = 16),
  legend.text = element_text(size = 14),
)
