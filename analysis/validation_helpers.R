# Function to order by max PMI
order_by_max_pmi <- function(confusion_data) {
  # Get ordering for actual (rows)
  row_order <- confusion_data %>%
    group_by(actual) %>%
    summarise(max_pmi = max(pmi, na.rm = TRUE)) %>%
    arrange(desc(max_pmi)) %>%
    pull(actual)
  
  # Get ordering for predicted (columns)
  col_order <- confusion_data %>%
    group_by(predicted) %>%
    summarise(max_pmi = max(pmi, na.rm = TRUE)) %>%
    arrange(desc(max_pmi)) %>%
    pull(predicted)
  
  confusion_data %>%
    mutate(
      actual = factor(actual, levels = row_order),
      predicted = factor(predicted, levels = col_order)
    )
}

# Plotting functions
plot_confusion_counts <- function(confusion_data, title, x_lab, y_lab) {
  ggplot(confusion_data, aes(x = predicted, y = actual, fill = count)) +
    geom_tile(color = "gray80", linewidth = 0.2) +
    geom_text(aes(label = count), color = "black", size = 3) +
    scale_fill_gradient(low = "white", high = "steelblue") +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
      plot.title = element_text(hjust = 0.5, face = "bold", size = 14)
    ) +
    labs(title = title, x = x_lab, y = y_lab, fill = "Count") +
    coord_fixed()
}

plot_confusion_proportions <- function(confusion_data, title, x_lab, y_lab) {
  ggplot(confusion_data, aes(x = predicted, y = actual, fill = prop)) +
    geom_tile(color = "gray80", linewidth = 0.2) +
    geom_text(aes(label = sprintf("%.2f", prop)), color = "black", size = 3) +
    scale_fill_gradient(low = "white", high = "steelblue") +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
      plot.title = element_text(hjust = 0.5, face = "bold", size = 14)
    ) +
    labs(title = title, x = x_lab, y = y_lab, fill = "Proportion") +
    coord_fixed()
}

plot_confusion_pmi <- function(confusion_data, title, x_lab, y_lab) {
  ggplot(confusion_data, aes(x = predicted, y = actual, fill = pmi)) +
    geom_tile(color = "gray80", linewidth = 0.2) +
    scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
      plot.title = element_text(hjust = 0.5, face = "bold", size = 14)
    ) +
    labs(title = title, x = x_lab, y = y_lab, fill = "PMI") +
    coord_fixed()
}

# Function to add proportions and PMI with Laplace smoothing
# min count = min number of exemplars sampled to include in plot
add_proportions_and_pmi <- function(confusion_data, all_activities, laplace = 1, min_count = 3) {
  # Ensure all activities are present in both dimensions
  confusion_complete <- confusion_data %>%
    complete(
      actual = all_activities, 
      predicted = all_activities, 
      fill = list(count = 0)
    ) %>%
    mutate(count_smoothed = count + laplace)
  
  # Calculate proportions and PMI
  total <- sum(confusion_complete$count_smoothed)
  
  result <- confusion_complete %>%
    group_by(actual) %>%
    mutate(
      row_total = sum(count_smoothed),
      prop = count_smoothed / row_total,
      keep_row = sum(count) >= min_count
    ) %>%
    ungroup() %>%
    filter(!predicted %in% actual[!keep_row]) %>%
    group_by(predicted) %>%
    mutate(col_total = sum(count_smoothed)) %>%
    ungroup() %>%
    mutate(
      p_actual = row_total / total,
      p_predicted = col_total / total,
      p_joint = count_smoothed / total,
      pmi = log2(p_joint / (p_actual * p_predicted))
    ) %>%
    filter(keep_row) %>%
    select(actual, predicted, count, prop, pmi)
  return(result)
}

# Function to create confusion matrix with counts
create_confusion_data <- function(data) {
  data %>%
    filter(!is.na(predicted) & !is.na(actual)) %>%
    group_by(actual, predicted) %>%
    summarise(count = n(), .groups = "drop")
}
