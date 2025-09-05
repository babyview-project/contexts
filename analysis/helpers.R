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

location_color <- "#6B8E23"
activity_color <- "#f6b26b"
activity_palette <- ggthemes::solarized_pal()(8)
top_activities <- c("cleaning", "cooking", "eating", "exploring", "looking at device", 
                    "playing", "reading", "watching tv")

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
