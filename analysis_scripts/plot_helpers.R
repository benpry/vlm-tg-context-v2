theme_set(theme_bw(base_size = 13) +
  theme(
    panel.grid = element_blank(),
    strip.background = element_blank()
  ))

strip_zero <- function(x, dp = 2) {
  sprintf(glue("%.{dp}f"), x) |>
    str_replace("0\\.", "\\.")
}

COL_SCALE <- scale_color_manual(
  values = c(
    "human_original" = "#f56942",
    "human_naive" = "#e89f46",
    "model_qwen" = "#32d97a",
    "model_gemma" = "#2cd9e6",
    "model_llama" = "#4388e8",
    "model_kimi" = "#b36ce3"
  ),
  labels = c(
    "human_original" = "Human (original)",
    "human_naive" = "Human (naïve)",
    "model_qwen" = "Qwen 2.5 VL",
    "model_gemma" = "Gemma 3",
    "model_llama" = "Llama 3.2",
    "model_kimi" = "Kimi VL"
  ),
  limits = c(
    "human_original",
    "human_naive",
    "model_qwen",
    "model_gemma",
    "model_llama",
    "model_kimi"
  )
)

COL_SCALE_NOLIM <- scale_color_manual(
  values = c(
    "human_original" = "#f56942",
    "human_naive" = "#e89f46",
    "model_qwen" = "#32d97a",
    "model_gemma" = "#2cd9e6",
    "model_llama" = "#4388e8",
    "model_kimi" = "#b36ce3"
  ),
  labels = c(
    "human_original" = "Human (original)",
    "human_naive" = "Human (naïve)",
    "model_qwen" = "Qwen 2.5 VL",
    "model_gemma" = "Gemma 3",
    "model_llama" = "Llama 3.2",
    "model_kimi" = "Kimi VL"
  )
)

condition_order <- c(
  "yoked", "shuffled", "backward", "ablated",
  "other-within", "other-across", "random", "no context"
)

make_accuracy_plot <- function(df, repnum_type = "matcher",
                               ref_level = 1 / 12) {
  p <- df |>
    mutate(
      repnum = if (repnum_type == "original") orig_repNum + 1 else matcher_repNum + 1,
      trialnum = if (repnum_type == "original") orig_trialNum + 1 else matcher_trialNum + 1,
      condition = factor(condition, levels = condition_order)
    ) |>
    ggplot(aes(x = repnum, y = accuracy, col = type)) +
    geom_hline(yintercept = ref_level, lty = "dashed") +
    # geom_point(position = position_jitter(width = .2), alpha = .05) +
    # stat_summary(
    #   aes(group = interaction(gameId, type, trialnum)),
    #   fun = mean, geom = "point", alpha = .01,
    #   position = position_jitter(width = .2),
    #   show.legend = TRUE
    # ) +
    stat_summary(
      aes(group = interaction(gameId, type)),
      fun = mean, geom = "line", alpha = .15,
      show.legend = TRUE
    ) +
    # geom_smooth(method = "glm", formula = y ~ log(x)) +
    geom_smooth(
      method = "loess", formula = y ~ x, se = FALSE,
      show.legend = TRUE
    ) +
    stat_summary(
      fun.data = mean_cl_boot, geom = "pointrange",
      show.legend = TRUE
    ) +
    scale_x_continuous(breaks = 1:6) +
    COL_SCALE +
    labs(
      x = glue("{str_to_sentence(repnum_type)} repetition number"),
      y = "Accuracy", col = "Matcher"
    )

  if (n_distinct(df$condition) > 1) {
    p <- p +
      facet_wrap(
        ~condition,
        nrow = 2,
        labeller = as_labeller(str_to_sentence)
      )
  }

  p
}
